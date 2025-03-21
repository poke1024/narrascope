from __future__ import annotations

import click
import json
import pickle
import numpy as np
import re
import collections
import base64
import datetime
import traceback
import logging
import sys
import multiprocessing

from pathlib import Path
from tqdm import tqdm
from itertools import groupby, chain
from functools import reduce, cache, partial
from intervaltree import Interval, IntervalTree

SCHEMA_VERSION = "2024-11-29"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Video:
	def __init__(self, path):
		pass


def export_video(p, catalog_path):
	if catalog_path:
		catalog = SimpleCatalogV2.load(catalog_path)
	else:
		catalog = FallbackCatalog()
		
	video = Video(p)
	try:
		return video.export(catalog)
	except Exception as e:
		logging.error(f"failed to process {p}")
		logging.exception(e)
		return None


def gather_files(p: Path, files):
	for q in p.iterdir():
		if q.is_dir():
			if (q / "transnet_shotdetection.pkl").exists():
				files.append(q)
			else:
				gather_files(q, files)
	


class Corpus:
	def __init__(self, files):
		self.files = files

	@staticmethod
	def read(base_path: Path):
		files = []
		gather_files(base_path, files)
		if not files:
			raise RuntimeError(f"failed to find any files under {base_path}")
		return Corpus(files)

	def export(self, out_path: Path, catalog_path: Path, limit=None, processes=1):
		video_data = []

		if limit is None or limit < 0:
			limit = len(self.files)

		files = self.files[:limit]

		with multiprocessing.Pool(processes=processes) as pool:
			with tqdm(total=len(files)) as pbar:
				f = partial(export_video, catalog_path=catalog_path)
				for data in pool.imap_unordered(f, files):
					if data is not None:
						video_data.append(data)
					pbar.update(1)

		with open(out_path, "w") as f:
			f.write(json.dumps({
				"version": SCHEMA_VERSION,
				"videos": video_data
			}, sort_keys=True, indent=4))


class FramewiseShotScaleClassification:
	def __init__(self, path):
		with open(path / "camera_size_classification.pkl", "rb") as f:
			data = pickle.load(f)

		self.index = [
			re.sub("^p_", "", x)
			for x in data["output_data"]["index"]]

		self.full_names = {
			"ECU": "EXTREME_CLOSE_UP",
			"CU": "CLOSE_UP",
			"MS": "MEDIUM_SHOT",
			"FS": "FULL_SHOT",
			"LS": "LONG_SHOT"
		}

		dt = data["output_data"]["delta_time"]

		self.tree = IntervalTree([
			Interval(t, t + dt, y)
			for t, y in zip(
				data["output_data"]["time"],
				data["output_data"]["y"])
		])

	def query(self, t0, t1):
		y = np.sum([iv.data for iv in self.tree.overlap(t0, t1)], axis=0)
		return self.full_names[self.index[np.argmax(y)]]


class Box:
	def __init__(self, p0, p1):
		self.p0 = np.array(p0, dtype=np.float64)
		self.p1 = np.array(p1, dtype=np.float64)

	@staticmethod
	def xx(x0, x1):
		return Box([x0], [x1])

	@staticmethod
	def from_bbox(data):
		return Box(
			[data["x"], data["y"]],
			[data["x"] + data["w"], data["y"] + data["h"]])

	@staticmethod
	def intersect(a: Box, b: Box):
		return Box(
			np.maximum(a.p0, b.p0),
			np.minimum(a.p1, b.p1))

	@staticmethod
	def iou(a: Box, b: Box):
		x = Box.intersect(a, b).area
		return x / (a.area + b.area - x)

	def coverage(self, a: Box):
		x = Box.intersect(self, a)
		return x.area / self.area

	@property
	def area(self):
		return np.prod(np.maximum(self.p1 - self.p0, 0))

	def __repr__(self):
		return f"Box({self.p0}, {self.p1})"


def face_size(xs):
	return np.median([x["bbox"]["h"] for x in xs])


def is_actor(xs):
	# copied from https://github.com/FakeNarratives/fakenarratives/blob/b328665d865d26a2808e6e2974674eb5555c1fba/graph_visualization/export_graphml.py#L220
	# FIXME: better implementation
	return bool(np.median([x["bbox"]["h"] for x in xs]) > 0.1)


def to_percentage(x):
	return float(round(x, 3))


def to_speaker_turns(segments):
	prep = map(lambda x: {
		"start": x["start"],
		"end": x["end"],
		"speaker": x.get("speaker") or "UNKNOWN",
		"words": x["text"].split()
	}, sorted(segments, key=lambda x: x["start"]))

	for sp, xs in groupby(prep, key=lambda x: x["speaker"]):
		xs = list(xs)
		yield {
			"start": xs[0]["start"],
			"end": xs[-1]["end"],
			"speaker": sp,
			"words": list(chain(*[x["words"] for x in xs]))
		}


def top_of_p_list(xs, min_p=0.5):
	if xs:
		best = max(xs, key=lambda x: x["p"])
		if best["p"] > min_p:
			return best["name"]

	return "undefined"


def p_list(keys, threshold, k_filter=None):
	if k_filter is None:
		k_filter = lambda k: k.lower()

	def make(probs):
		r = []
		for k, v in zip(keys, probs):
			if v >= threshold:
				r.append({"name": k_filter(k), "p": to_percentage(v)})
		return r

	return make


def keyed_p_list(threshold):
	def make(kv):
		r = []
		for k, v in kv.items():
			if v >= threshold:
				r.append({"name": k.lower(), "p": to_percentage(v)})
		return r

	return make


class Gaze:
	def __init__(self, x):
		try:
			hv = np.mean([
				x["left_gaze_deg"],
				x["right_gaze_deg"]], axis=0)
		except Exception as e:
			logging.exception(e)
			hv = [0, 0]
		self.hv = hv

	@staticmethod
	def median(xs):
		hv = [round(x, 1) for x in np.median([
			x.hv for x in xs], axis=0)]
		return {
			"h": hv[0],
			"v": hv[1]
		}


def make_interval_tree(xs):
	return IntervalTree(list(filter(lambda x: x.begin < x.end, xs)))


def avg_emotions(xs):
	labels = [
		"angry",
		"surprise",
		"fear",
		"sad",
		"happy",
		"disgust",
		"neutral"
	]
	
	assert len(xs) >= 1
	ks = set(xs[0].keys())
	assert all(set(x.keys()) == ks for x in xs[1:])
	data = np.array([[x[k] for k in ks] for x in xs])
	avg_data = np.mean(data, axis=0)
	assert avg_data.shape[0] == len(ks)
	kv = dict(zip(ks, avg_data))
	return dict(([k, round(kv.get(k, 0), 1)] for k in labels))
	

class FaceData:
	def __init__(self, path):
		with open(path / "face_analysis.pkl", "rb") as f:
			data = pickle.load(f)

		if all("delta_time" in d for d in data["faces"]):
			self.tree = IntervalTree([
				Interval(d["time"], d["time"] + d["delta_time"], d)
				for d in data["faces"]])
		else:
			ivs = [(d0["time"], d1["time"], d0) for d0, d1 in zip(data["faces"], data["faces"][1:])]
			assert all(iv[0] is not None for iv in ivs)
			assert all(iv[1] is not None for iv in ivs)
			ivs = [iv for iv in ivs if iv[0] < iv[1]]
			self.tree = IntervalTree.from_tuples(ivs)

		self.regions = {
			'left': Box([0, 0], [0.5, 1]),
			'center': Box([0.25, 0], [0.75, 1]),
			'right': Box([0.5, 0], [1, 1])
		}

		# see https://github.com/FakeNarratives/fakenarratives/blob/b328665d865d26a2808e6e2974674eb5555c1fba/graph_visualization/export_graphml.py#L303
		self.headpose_keys = [
			"yaw",
			"pitch",
			"roll"
		]

	def query(self, t0, t1):
		assert t0 is not None
		assert t1 is not None

		num_faces = 0
		faces = []

		ivs = self.tree.overlap(t0, t1)

		for i, iv in enumerate(ivs):
			if iv.data["cluster_id"] is None or iv.data["cluster_id"] < 0:
				iv.data["cluster_id"] = -(1 + i)  # unique id

		key = lambda x: x.data["cluster_id"]
		for k, ivs in groupby(sorted(ivs, key=key), key=key):
			num_faces += 1

			xs = [x.data for x in ivs]

			screen_time = min(1., np.sum([x["delta_time"] for x in xs]) / (t1 - t0))
			if screen_time < 0.5:
				continue

			faces.append({
				"id": str(k),
				"size": face_size(xs),
				"emotion": avg_emotions([x["emotions"] for x in xs]),

				"stime": screen_time,

				"head": dict(yaw=0, pitch=0, roll=0),
				"gaze": Gaze.median([Gaze(x["gaze"]) for x in xs]),

				"speaking": any(x["speaking"] for x in xs),

				"region": dict(
					(box_k, to_percentage(np.sqrt(np.median([
						Box.coverage(box, Box.from_bbox(x["bbox"])) for x in xs]))))
					for box_k, box in self.regions.items())
			})

		return num_faces, faces


def weighted_prob(tree, t0, t1, k_prob):
	a = []
	w = []

	for iv in tree.overlap(t0, t1):
		a.append(iv.data[k_prob])
		w.append(Box.coverage(Box.xx(iv.data["start"], iv.data["end"]), Box.xx(t0, t1)))

	if not w or sum(w) == 0:
		return []

	return np.average(a, weights=w, axis=0)


def listify(x):
	if isinstance(x, list):
		return x
	else:
		return [x]


def weighted_prob_k(tree, t0, t1, k_label, k_prob):
	r = []
	w = []

	for iv in tree.overlap(t0, t1):
		r.append(dict(zip(
			listify(iv.data[k_label]),
			listify(iv.data[k_prob]))))
		w.append(Box.coverage(Box.xx(iv.data["start"], iv.data["end"]), Box.xx(t0, t1)))

	if not w or sum(w) == 0:
		return dict()

	k2i = dict((k, i) for i, k in enumerate(set(chain(*[x.keys() for x in r]))))
	i2k = [x[0] for x in sorted(k2i.items(), key=lambda x: x[1])]

	a = np.zeros((len(r), len(i2k)), dtype=np.float64)
	for i, x in enumerate(r):
		for k, p in x.items():
			a[i, k2i[k]] = p

	return dict((k, v) for k, v in zip(i2k, np.average(a, weights=w, axis=0)))


def weighted_mode(tree, t0, t1, k_data, default):
	w = collections.defaultdict(lambda: 0.)

	for iv in tree.overlap(t0, t1):
		w[iv.data[k_data]] += Box.coverage(
			Box.xx(iv.data["start"], iv.data["end"]),
			Box.xx(t0, t1))

	if not w:
		return default

	return max(w.items(), key=lambda x: x[1])[0]


class SpeakerSentimentData:
	def __init__(self, path):
		with open(path / "whisperx_sentiment.pkl", "rb") as f:
			data = pickle.load(f)

		ws = dict(
			neutral=0,
			positive=1,
			negative=-1)

		self.tree = make_interval_tree([
			Interval(d["start"], d["end"], ws[d["pred"] or "neutral"])
			for d in data["output_data"]["model_news"]["speakerturn_wise"]])

	def query(self, t0, t1):
		ys = [iv.data for iv in self.tree.overlap(t0, t1)]
		if len(ys) < 1:
			return "neutral"
		y_mean = np.mean(ys)
		if abs(y_mean) < 0.5:
			return "neutral"
		elif y_mean > 0:
			return "positive"
		else:
			return "negative"


class SpeakerWordClassData:
	def __init__(self, path):
		with open(path / "whisperx_pos.pkl", "rb") as f:
			data = pickle.load(f)

		self.tags = [
			"verb",
			"propn",
			"pron",
			"part",
			"num",
			"noun",
			"intj",
			"det",
			"conj",
			"aux",
			"adv",
			"adp",
			"adj"
		]

		tag2i = dict((k, i) for i, k in enumerate(self.tags))

		def counts_vector(d):
			v = np.zeros((len(self.tags),), dtype=np.int32)
			for x in (d.get("tags") or []):
				index = tag2i.get(x[1].lower())
				if index is not None:
					v[index] += 1
			return v

		self.tree = make_interval_tree([
			Interval(d["start"], d["end"], counts_vector(d))
			for d in data["output_data"]["speakerturn_wise"]])

	def query(self, t0, t1):
		ys = []
		for iv in self.tree.overlap(t0, t1):
			ys.append(iv.data)

		if len(ys) > 0:
			return dict(zip(self.tags, np.sum(ys, axis=0).tolist()))
		else:
			return dict(zip(self.tags, [0] * len(self.tags)))


class SpeakerAudioClfData:
	def __init__(self, path):
		with open(path / "whisperxspeaker_audioClf.pkl", "rb") as f:
			data = pickle.load(f)

		self.tree = make_interval_tree([
			Interval(d["start"], d["end"], d)
			for d in data["output_data"]])

		self.p_list = keyed_p_list(0.5)

	def query(self, t0, t1):
		x = weighted_prob_k(self.tree, t0, t1, "top3_label", "top3_label_prob")
		return self.p_list(x)


class SpeakerSegmentClfData:
	def __init__(self, path):
		with open(path / "whisperxspeaker_segmentClf.pkl", "rb") as f:
			data = pickle.load(f)

		self.tree = make_interval_tree([
			Interval(d["start"], d["end"], d)
			for d in data["output_data"]])

		self.p_list = keyed_p_list(0.5)

	def emotion(self, t0, t1):
		x = weighted_prob_k(self.tree, t0, t1, "emotion_pred_top3", "emotion_prob_top3")
		return self.p_list(x)

	def gender(self, t0, t1):
		return weighted_mode(self.tree, t0, t1, "gender_pred", "UNKNOWN_GENDER").upper()


def load_sim_data(path, method, feature, suffix, value, default):
	full_path = path / f"{method}{suffix}.pkl"
	
	if not full_path.exists():
		logging.warning(f"failed to find {full_path}. ignoring.")
		return EmptyShotRecords(default)

	with open(full_path, "rb") as f:
		data = pickle.load(f)

	try:
		ivs = []
		for x in data["output_data"]:
			t0 = x["shot"]["start"]
			t1 = x["shot"]["end"]
			y = x.get(feature)
			if y is None:
				z = default
			else:
				z = value(y)
			ivs.append((t0, t1, round(float(z), 2)))
		return ShotRecords(ivs)
	except:
		raise RuntimeError(f"failed to parse {full_path}")


class ShotSimData:
	def __init__(self, path, feature, methods, suffix, value, default):
		self.data = dict((k, load_sim_data(
			path, k, feature, suffix, value, default)) for k in methods)

	def query(self, t0, t1):
		return dict(
			(k.lower().replace("-", ""), v.query(t0, t1))
			for k, v in self.data.items())


def mode(xs):
	return collections.Counter(xs).most_common(1)[0][0]


def fix_iv(iv):
	if iv[0] == iv[1]:
		return (iv[0], iv[0] + 0.01, iv[2])
	else:
		return iv


class EmptyShotRecords:
	def __init__(self, default):
		self.default = default
	
	def query(self, t0, t1):
		return self.default


class BestEffortShotRecords:
	def __init__(self, ivs):
		ivs = [fix_iv(iv) for iv in ivs]
		self.tree = IntervalTree.from_tuples(ivs)

	def query(self, t0, t1):
		xs = list(self.tree.overlap(t0, t1))
		if not xs:
			raise RuntimeError("no overlap")
		q = Box.xx(t0, t1)
		ys = [q.coverage(Box.xx(x.begin, x.end)) for x in xs]
		return xs[np.argmax(ys)].data


class ExactShotRecords:
	def __init__(self, ivs):
		self.data = dict()
		for t0, t1, x in ivs:
			self.data[(t0, t1)] = x

	def query(self, t0, t1):
		return self.data[(t0, t1)]


ShotRecords = ExactShotRecords


class ShotAngleData:
	def __init__(self, path):
		with open(path / "videoshot_angle.pkl", "rb") as f:
			self.shot_angle_data = pickle.load(f)

		ivs = []
		for x in self.shot_angle_data["output_data"]:
			ivs.append((
				x["shot"]["start"],
				x["shot"]["end"],
				mode(x["predictions"]).upper()))
		self.records = ShotRecords(ivs)

	def query(self, t0, t1):
		return self.records.query(t0, t1)


class ShotLevelData:
	def __init__(self, path):
		with open(path / "videoshot_level.pkl", "rb") as f:
			self.shot_angle_data = pickle.load(f)

		ivs = []
		for x in self.shot_angle_data["output_data"]:
			ivs.append((
				x["shot"]["start"],
				x["shot"]["end"],
				mode(x["predictions"]).upper()))
		self.records = ShotRecords(ivs)

	def query(self, t0, t1):
		return self.records.query(t0, t1)


class ShotScaleMovementData:
	def __init__(self, path):
		with open(path / "videoshot_scalemovement.pkl", "rb") as f:
			self.shot_angle_data = pickle.load(f)

		self._scale = {}
		self._movement = {}

		full_scale_names = {
			"ECS": "EXTREME_CLOSE_UP",
			"CS": "CLOSE_UP",
			"MS": "MEDIUM_SHOT",
			"FS": "FULL_SHOT",
			"LS": "LONG_SHOT"
		}

		ivs = []
		for x in self.shot_angle_data["output_data"]:
			p = x["prediction"]			
			scale, cumulative_distance, movement = p

			ivs.append((
			x["shot"]["start"],
			x["shot"]["end"], {
				"scale": full_scale_names[scale.upper()],
				"cumulative_distance": cumulative_distance,
				"movement": movement.upper()
			}))
					
		self.records = ShotRecords(ivs)

	def scale(self, t0, t1):
		return self.records.query(t0, t1)["scale"]
		
	def cumulative_distance(self, t0, t1):
		return self.records.query(t0, t1)["cumulative_distance"]

	def movement(self, t0, t1):
		return self.records.query(t0, t1)["movement"]


class VLMData:
	def __init__(self, path, kind, renames=None, labels=[]):
		with open(path / f"vlm_{kind}.pkl", "rb") as f:
			data = pickle.load(f)
			
		out = data["output_data"]

		dt = out["delta_time"]
		ts = np.array(out["times"])
		ys = out["responses"].T

		self.tree = IntervalTree([
			Interval(t, t + dt, y)
			for t, y in zip(ts, ys)
		])

		self.labels = labels
		
		out_labels = out["labels"]
		if renames:
			out_labels = [renames.get(x, x) for x in out_labels]
		self.out_index = dict((k, i) for i, k in enumerate(out_labels))
		
	def _x(self, ys):
		zs = []
		for k in self.labels:
			i = self.out_index.get(k)
			if i is None:
				zs.append(0)
			else:
				zs.append(ys[i])
		return zs
		

	def query(self, t0, t1):
		ys = [iv.data for iv in self.tree.overlap(t0, t1)]
		if len(ys) == 0:
			return dict(zip(self.labels, [0] * len(self.labels)))
		else:
			mean_y = np.mean(ys, axis=0)
			return dict(zip(self.labels, self._x(mean_y)))


class EntitiesData:
	def __init__(self, path):
		with open(path / f"whisperx_ner.pkl", "rb") as f:
			data = pickle.load(f)

		self.tree = IntervalTree()

		self.labels = [x.lower() for x in data["output_data"]["ner_labelmap"].keys()]
		ivs = []

		for record in data["output_data"]["speakerturn_wise"]:
			t0 = record["start"]
			t1 = record["end"]
			t1 = max(t1, t0 + 0.01)  # prevent null intervals

			counts = collections.Counter(dict(zip(
				self.labels, [0] * len(self.labels))))
			for tag in (record.get("tags") or []):
				counts[tag["type"].lower()] += 1

			ivs.append(Interval(t0, t1, np.array(
				[counts[k] for k in self.labels], dtype=np.int32)))

		self.tree = IntervalTree(ivs)

	def query(self, t0, t1):
		ys = []

		for iv in self.tree.overlap(t0, t1):
			ys.append(iv.data)

		if len(ys) > 0:
			return dict(zip(self.labels, np.sum(ys, axis=0).tolist()))
		else:
			return dict(zip(self.labels, [0] * len(self.labels)))


class ShotData:
	def __init__(self, path):
		with open(path / "transnet_shotdetection.pkl", "rb") as f:
			self.shot_detection_data = pickle.load(f)

		self.path = path
		self.instruct_blip = True

	def iter(self):
		# shot_scale_classification = FramewiseShotScaleClassification(self.path)
		face_data = FaceData(self.path)
		speaker_audio_clf = SpeakerAudioClfData(self.path)
		shot_angle_data = ShotAngleData(self.path)
		shot_level_data = ShotLevelData(self.path)
		scale_movement_data = ShotScaleMovementData(self.path)
		
		if self.instruct_blip:
			place_data = VLMData(self.path, "locations", renames={
				"news studio": "studio"
			}, labels=[
				'indoor',
				'rural',
				'industrial',
				'urban',
				'suburban',
				'studio'])
			roles_data = VLMData(self.path, "social_roles", labels=[
				'protester',
				'anchor',
				'reporter',
				'politician',
				'doctor',
				'layperson'])
		else:
			place_data = None
			roles_data = None

		shot_sim = {
			"visual": {
				"methods": ["siglip", "convnextv2", "places"],
				"suffix": "_shot_similarity",
				"value": lambda x: x[1]  # median
			},
			"action": {
				"methods": ["kinetics-vmae", "ssv2-vmae", "kinetics-xclip"],
				"suffix": "_action_shot_similarity",
				"value": lambda x: x
			},
			"audio": {
				"methods": ["wav2vec2", "beats"],  #, "whisper"],
				"suffix": "_audio_shot_similarity",
				"value": lambda x: x
			}
		}

		shot_sim_data = dict()
		for domain, args in shot_sim.items():
			r = dict()
			for scope in ["next_1", "next_2"]:
				r[scope] = ShotSimData(self.path, scope, **args, default=0)
			shot_sim_data[domain] = r

		def query_sim(t0, t1, k):
			r = dict()
			for domain, scopes in shot_sim_data.items():
				r[domain] = scopes[k].query(t0, t1)
			return r

		for i, shot_rec in enumerate(self.shot_detection_data["output_data"]["shots"]):
			t0 = float(shot_rec["start"])
			t1 = float(shot_rec["end"])

			num_faces, faces = face_data.query(t0, t1)

			yield {
				"index": i,
				"startTime": t0,
				"endTime": t1,
				"scale": scale_movement_data.scale(t0, t1),
				"angle": shot_angle_data.query(t0, t1),
				"level": shot_level_data.query(t0, t1),
				"movement": scale_movement_data.movement(t0, t1),
				"cumulativeDistance": scale_movement_data.cumulative_distance(t0, t1),
				"people": num_faces,
				"faces": faces,
				"tags": speaker_audio_clf.query(t0, t1),
				"next1": query_sim(t0, t1, "next_1"),
				"next2": query_sim(t0, t1, "next_2"),
				"place": place_data.query(t0, t1) if place_data else [],
				"roles": roles_data.query(t0, t1)
			}


class SpeakerTurnMetaData:
	def __init__(self, path):
		with open(path / "speaker_turns_meta.pkl", "rb") as f:
			data = pickle.load(f)

		ivs = collections.defaultdict(list)
		for x in data["output_data"]:
			ivs[x["speaker"]].append((x["start"], x["end"], x))

		self.tree = {k: IntervalTree.from_tuples(
			[x for x in v if x[0] < x[1]]) for k, v in ivs.items()}

	def query(self, t0, t1, speaker):
		records = []
		
		if speaker in self.tree:
			for iv in self.tree[speaker].overlap(t0, t1):
				r1 = max(t0, iv.begin)
				r2 = min(t1, iv.end)
				if r1 < r2:
					records.append((r2 - r1, iv.data))

		if not records:
			return {
				'active_ratio': 0,
				'role_l0': 'unsure',
				'situation': 'unsure'
			}
		else:
			i = np.argmax([x[0] for x in records])
			return records[i][1]



class SpeakerTurnLabelData:
	def __init__(self, path, name, extract):
		with open(path / name, "rb") as f:
			data = pickle.load(f)

		ivs = collections.defaultdict(list)
		for x in data["output_data"]:
			ivs[x["speaker"]].append((x["start"], x["end"], extract(x["label"])))

		self.tree = {k: IntervalTree.from_tuples(
			[x for x in v if x[0] < x[1]]) for k, v in ivs.items()}

	def query(self, t0, t1, speaker):
		if speaker in self.tree:
			xs = [iv.data for iv in self.tree[speaker].overlap(t0, t1)]
			return np.mean(xs) if len(xs) > 0 else 0
		else:
			return 0


class SpeakerTurnData:
	def __init__(self, path):
		with open(path / "asr_whisperx.pkl", "rb") as f:
			data = pickle.load(f)

		self.turns = to_speaker_turns(
			data["output_data"]["speaker_segments"])

		self.path = path

	def iter(self):
		sentiment2_to_value = {
			'very negative': -1,
			'neutral': 0,
			'very positive': 1,
			'None': 0
		}

		speaker_sentiment = SpeakerSentimentData(self.path)
		speaker_word_class = SpeakerWordClassData(self.path)
		speaker_audio_clf = SpeakerAudioClfData(self.path)
		speaker_segment_clf = SpeakerSegmentClfData(self.path)
		ent_data = EntitiesData(self.path)
		meta_data = SpeakerTurnMetaData(self.path)
		evaluative_data = SpeakerTurnLabelData(
			self.path,
			"llm_evaluative.pkl",
			lambda x: 1 if x == "evaluative" else 0)
		sentiment_2_data = SpeakerTurnLabelData(
			self.path,
			"llm_sentiment_2.pkl",
			lambda x: sentiment2_to_value[x])

		for turn in self.turns:
			t0 = float(turn["start"])
			t1 = float(turn["end"])
			emotions = speaker_segment_clf.emotion(t0, t1)
			meta = meta_data.query(t0, t1, turn["speaker"])
			yield {
				"id": turn["speaker"],
				"startTime": t0,
				"endTime": t1,
				"numWords": len(turn["words"]),
				"words": speaker_word_class.query(t0, t1),
				"entities": ent_data.query(t0, t1),
				"sentiment": speaker_sentiment.query(t0, t1),
				"tags": speaker_audio_clf.query(t0, t1),
				"gender": speaker_segment_clf.gender(t0, t1),
				"emotion": top_of_p_list(emotions),
				"active": meta['active_ratio'],
				"role": meta['role_l0'],
				"situation": meta['situation'],
				"evaluative": evaluative_data.query(t0, t1, turn["speaker"]),
				"sentiment2": sentiment_2_data.query(t0, t1, turn["speaker"])
			}


class Catalog:
	def get(self, path: Path):
		raise NotImplementedError()


class SimpleCatalogV1(Catalog):
	def get(self, path: Path):
		if path.parent.name == "Tagesschau":
			m = re.search(r"TV-(\d+)-(\d+)-(\d+)", path.stem)
			if m is not None:
				year = int(m.group(1)[:4])
				month = int(m.group(1)[4:6])
				day = int(m.group(1)[6:8])
				return {
					"filename": path.name,
					"channel": "Tagesschau",
					"title": f"Sendung vom {day}.{month}.{year}",
					"publishedAt": datetime.datetime(year, month, day).timestamp(),
					"url": "",
					"id": f"Tagesschau/{re.sub(r'^whisper_', '', path.stem)}"
				}

		if path.parent.name == "BildTV":
			m = re.match(r"(\d{4})(\d{2})(\d{2})_.+", path.stem)
			if m is not None:
				year = int(m.group(1))
				month = int(m.group(2))
				day = int(m.group(3))
				return {
					"filename": path.name,
					"channel": "BildTV",
					"title": f"BildTV vom {day}.{month}.{year}",
					"publishedAt": datetime.datetime(year, month, day).timestamp(),
					"url": "",
					"id": f"BildTV/{re.sub(r'^whisper_', '', path.stem)}"
				}
			else:
				raise RuntimeError(f"failed to parse name {path.stem}")

		if path.parent.name == "CompactTV":
			m = re.match(r"compacttv_(\d{4})_(\d{2})_(\d{2})_.+", path.stem)
			if m is not None:
				year = int(m.group(1))
				month = int(m.group(2))
				day = int(m.group(3))
				return {
					"filename": path.name,
					"channel": "CompactTV",
					"title": f"Sendung vom {day}.{month}.{year}",
					"publishedAt": datetime.datetime(year, month, day).timestamp(),
					"url": "",
					"id": f"CompactTV/{re.sub(r'^whisper_', '', path.stem)}"
				}
			m = re.match(r"COMPACTTV-(\d{4})-(\d{2})-(\d{2})-.+", path.stem)
			if m is not None:
				year = int(m.group(1))
				month = int(m.group(2))
				day = int(m.group(3))
				return {
					"filename": path.name,
					"channel": "CompactTV",
					"title": f"Sendung vom {day}.{month}.{year}",
					"publishedAt": datetime.datetime(year, month, day).timestamp(),
					"url": "",
					"id": f"CompactTV/{re.sub(r'^whisper_', '', path.stem)}"
				}
			else:
				raise RuntimeError(f"failed to parse name {path.stem}")

		if path.parent.name == "HeuteJournal":
			m = re.match(r"(\d{4})-(\d{2})-(\d{2})T.+", path.stem)
			if m is not None:
				year = int(m.group(1))
				month = int(m.group(2))
				day = int(m.group(3))
				return {
					"filename": path.name,
					"channel": "Heute Journal",
					"title": f"Sendung vom {day}.{month}.{year}",
					"publishedAt": datetime.datetime(year, month, day).timestamp(),
					"url": "",
					"id": f"HeuteJournal/{re.sub(r'^whisper_', '', path.stem).replace('/', ':')}"
				}
			else:
				raise RuntimeError(f"failed to parse name {path.stem}")
		return None
		
		
class FallbackCatalog(Catalog):
	def get(self, path: Path):
		return {
			"filename": path.name,
			"channel": path.name,
			"title": path.name,
			"publishedAt": int(datetime.datetime.today().timestamp()),
			"url": "",
			"id": path.name
		}
		
		
class SimpleCatalogV2(Catalog):
		def __init__(self, records):
			self.records = records

		@staticmethod
		@cache
		def load(p: Path):
			records = dict()
			with open(p, "r") as f:
				for r in json.loads(f.read()):
					records[r["filename"].split(".")[0]] = r
			return SimpleCatalogV2(records)
	
		def get(self, path: Path):
			if path.parent.name == "Tagesschau":
				m = re.search(r"TV-(\d+)-(\d+)-(\d+)", path.stem)
				if m is not None:
					year = int(m.group(1)[:4])
					month = int(m.group(1)[4:6])
					day = int(m.group(1)[6:8])
					return {
						"filename": path.name,
						"channel": "Tagesschau",
						"title": f"Sendung vom {day}.{month}.{year}",
						"publishedAt": datetime.datetime(year, month, day).timestamp(),
						"url": "",
						"id": f"Tagesschau/{re.sub(r'^whisper_', '', path.stem)}"
					}
	
			r = self.records.get(path.stem)
			if r:
				return r
			else:
				raise RuntimeError(f"failed to parse name {path.stem}")
		
			return None


def encode_video_id(s):
	return base64.b64encode(s.encode("utf8")).decode("ascii")


class Video:
	def __init__(self, path):
		self.path = path
		self.preflight = False

	def export(self, catalog):
		meta = catalog.get(self.path)

		if meta is None:
			return None

		if self.preflight:
			return {}

		shot_data = ShotData(self.path)
		shots = list(shot_data.iter())

		speaker_turn_data = SpeakerTurnData(self.path)
		speaker_turns = list(speaker_turn_data.iter())

		return {
			"filename": meta["filename"],
			"channel": meta["channel"],
			"title": meta["title"],
			"publishedAt": meta["publishedAt"],
			"videoId": encode_video_id(meta["id"]),
			"shots": shots,
			"speakerTurns": speaker_turns
		}


@click.command()
@click.argument('pkl', type=click.Path(exists=True, file_okay=False))
@click.argument('out', type=click.Path(exists=False))
@click.option('--catalog', type=click.Path(exists=True), required=False)
@click.option('--limit', type=int, default=-1)
@click.option('--processes', type=int, default=1)
def tib2nar(pkl, out, catalog, limit: int, processes: int):
	"""Convert directory of TIB files (PKL) to narrascope JSON data file (OUT)."""
	out = Path(out)
	if not out.suffix == ".json":
		raise click.BadArgumentUsage("output file must end in .json")

	if not out.parent.exists():
		raise click.BadArgumentUsage(f"output path does not exist: {out.parent}")

	try:
		corpus = Corpus.read(Path(pkl))
		corpus.export(out, catalog, limit, processes=processes)
	except:
		traceback.print_exc()
	else:
		print(f"successfully exported {out}")


if __name__ == '__main__':
	tib2nar()

