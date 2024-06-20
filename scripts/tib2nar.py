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

from pathlib import Path
from tqdm import tqdm
from itertools import groupby, chain
from functools import reduce
from intervaltree import Interval, IntervalTree


SCHEMA_VERSION = "2024-06-20"


class Video:
	def __init__(self, path):
		pass

class Corpus:
	def __init__(self, files):
		self.files = files
	
	@staticmethod
	def read(base_path: Path):
		files = []
	
		for channel_path in base_path.iterdir():
			if not channel_path.name.startswith("."):
				for video_path in channel_path.iterdir():
					if (video_path / "transnet_shotdetection.pkl").exists():
						files.append(video_path)

		return Corpus(files)

	def export(self, out_path: Path, limit=None):
		catalog = SimpleCatalog()
		video_data = []

		if limit is None:
			limit = len(self.files)
		
		for p in tqdm(self.files[:limit]):
			video = Video(p)
			try:
				video_data.append(video.export(catalog))
			except Exception as e:
				print(e)
				raise click.ClickException(f"failed to process {p}")
	
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


def make_head_gaze(data):
	return {
		"h": float(round((data[2, 0] + data[3, 0]) / 2, 1)),
		"v": float(round((data[2, 1] + data[3, 1]) / 2, 1))
	}


def make_interval_tree(xs):
	return IntervalTree(list(filter(lambda x: x.begin < x.end, xs)))
	

class FaceData:
	def __init__(self, path):
		with open(path / "face_analysis.pkl", "rb") as f:
			data = pickle.load(f)

		self.tree = IntervalTree([
			Interval(d["time"], d["time"] + d["delta_time"], d)
			for d in data["faces"]])
		
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

		# see https://github.com/FakeNarratives/fakenarratives/blob/b328665d865d26a2808e6e2974674eb5555c1fba/graph_visualization/export_graphml.py#L295C38-L295C43
		self.emotion_keys = [
			"angry",
			"disgust",
			"fear",
			"happy",
			"sad",
			"surprise",
			"neutral"
		]

		self.emotion_p_list = p_list(self.emotion_keys, 0.5)

	def query(self, t0, t1):
		key = lambda x: x.data["cluster_id"]
		for k, ivs in groupby(sorted(self.tree.overlap(t0, t1), key=key), key=key):
			xs = [x.data for x in ivs]

			screen_time = min(1., np.sum([x["delta_time"] for x in xs]) / (t1 - t0))
			if screen_time < 0.5:
				continue

			emotions = self.emotion_p_list(
				np.mean([x["emotion"] for x in xs], axis=0))
			
			yield {
				"id": str(k),
				"size": face_size(xs),
				"emotion": top_of_p_list(emotions),
				"head": dict((k, int(v)) for k, v in zip(
					self.headpose_keys,
					np.mean([x["headpose"] for x in xs], axis=0))),
				"gaze": make_head_gaze(np.median([x["headgaze"] for x in xs], axis=0)),
				"actor": is_actor(xs),
				"stime": to_percentage(screen_time),
				"region": dict(
					(box_k, to_percentage(np.sqrt(np.median([
						Box.coverage(box, Box.from_bbox(x["bbox"])) for x in xs]))))
					for box_k, box in self.regions.items())
			}


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
		with open(path / "whisper_sentiment.pkl", "rb") as f:
			data = pickle.load(f)

		self.tree = make_interval_tree([
			Interval(d["start"], d["end"], d)
			for d in data["output_data"]["speakerturn_wise"]])
				
		dmap = data["output_data"]["sent_labelmap"]
		self.label_map = dict([(v, k) for k, v in dmap.items()])

		self.p_list = p_list(
			[self.label_map[i] for i in range(len(self.label_map))],
			0.5,
			k_filter=lambda x: x.upper() + "_SENTIMENT")
	
	def query(self, t0, t1):
		y = weighted_prob(self.tree, t0, t1, "prob")
		return self.p_list(y)
			
	
class SpeakerWordClassData:
	def __init__(self, path):
		with open(path / "whisper_pos.pkl", "rb") as f:
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
		with open(path / "whisperspeaker_audioClf.pkl", "rb") as f:
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
		with open(path / "whisperspeaker_segmentClf.pkl", "rb") as f:
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


def load_sim_data(path, method, feature, suffix, value):
	full_path = path / f"{method}{suffix}.pkl"

	with open(full_path, "rb") as f:
		data = pickle.load(f)

	try:
		r = dict()
		for x in data["output_data"]:
			t0 = x["shot"]["start"]
			t1 = x["shot"]["end"]
			r[(t0, t1)] = round(float(value(x[feature])), 2)
	except:
		raise RuntimeError(f"failed to parse {full_path}")

	return r


class ShotSimData:
	def __init__(self, path, feature, methods, suffix, value):
		self.data = dict((k, load_sim_data(
			path, k, feature, suffix, value)) for k in methods)
	
	def query(self, t0, t1):
		return dict(
			(k.lower().replace("-", ""), v[(t0, t1)])
			for k, v in self.data.items())


def mode(xs):
	return collections.Counter(xs).most_common(1)[0][0]


class ShotAngleData:
	def __init__(self, path):
		with open(path / "videoshot_angle.pkl", "rb") as f:
			self.shot_angle_data = pickle.load(f)

		self.data = {}
		
		for x in self.shot_angle_data["output_data"]:
			self.data[(x["shot"]["start"], x["shot"]["end"])] = mode(x["predictions"]).upper()
	
	def query(self, t0, t1):
		return self.data[(t0, t1)]


class ShotLevelData:
	def __init__(self, path):
		with open(path / "videoshot_level.pkl", "rb") as f:
			self.shot_angle_data = pickle.load(f)

		self.data = {}
		
		for x in self.shot_angle_data["output_data"]:
			self.data[(x["shot"]["start"], x["shot"]["end"])] = mode(x["predictions"]).upper()
	
	def query(self, t0, t1):
		return self.data[(t0, t1)]


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

		for x in self.shot_angle_data["output_data"]:
			key = (x["shot"]["start"], x["shot"]["end"])
			scale, movement = x["prediction"]
			self._scale[key] = full_scale_names[scale.upper()]
			self._movement[key] = movement.upper()
	
	def scale(self, t0, t1):
		return self._scale[(t0, t1)]

	def movement(self, t0, t1):
		return self._movement[(t0, t1)]


class InstructBLIPData:
	def __init__(self, path, kind, renames=None):
		with open(path / f"instructblip_{kind}.pkl", "rb") as f:
			data = pickle.load(f)

		dt = data["delta_time"]
		ts = np.array(data["time"])
		ys = data["responses"].T

		self.tree = IntervalTree([
			Interval(t, t + dt, y)
			for t, y in zip(ts, ys)
		])

		self.labels = data["labels"]

		if renames:
			self.labels = [renames.get(x, x) for x in self.labels]

	def query(self, t0, t1):
		ys = [iv.data for iv in self.tree.overlap(t0, t1)]
		if len(ys) == 0:
			return dict(zip(self.labels, [0] * len(self.labels)))
		else:
			mean_y = np.mean(ys, axis=0)
			return dict(zip(self.labels, mean_y))


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

	def iter(self):
		#shot_scale_classification = FramewiseShotScaleClassification(self.path)
		face_data = FaceData(self.path)
		speaker_audio_clf = SpeakerAudioClfData(self.path)
		shot_angle_data = ShotAngleData(self.path)
		shot_level_data = ShotLevelData(self.path)
		scale_movement_data = ShotScaleMovementData(self.path)
		place_data = InstructBLIPData(self.path, "environment", renames={
			"news studio": "studio"
		})
		roles_data = InstructBLIPData(self.path, "news_roles")

		shot_sim = {
			"image": {
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
				"methods": ["wav2vec2", "beats", "whisper"],
				"suffix": "_audio_shot_similarity",
				"value": lambda x: x
			}
		}

		shot_sim_data = dict()
		for domain, args in shot_sim.items():
			r = dict()
			for scope in ["next_1", "next_2"]:
				r[scope] = ShotSimData(self.path, scope, **args)
			shot_sim_data[domain] = r

		def query_sim(t0, t1):
			r = dict()
			for domain, scopes in shot_sim_data.items():
				r[domain] = scopes["next_1"].query(t0, t1)
			return r
		
		for shot_rec in self.shot_detection_data["output_data"]["shots"]:
			t0 = float(shot_rec["start"])
			t1 = float(shot_rec["end"])
			yield {
				"startTime": t0,
				"endTime": t1,
				#"scale": shot_scale_classification.query(t0, t1),
				"scale": scale_movement_data.scale(t0, t1),
				"angle": shot_angle_data.query(t0, t1),
				"level": shot_level_data.query(t0, t1),
				"movement": scale_movement_data.movement(t0, t1),
				"faces": list(face_data.query(t0, t1)),
				"tags": speaker_audio_clf.query(t0, t1),
				"next": query_sim(t0, t1),
				"place": place_data.query(t0, t1),
				"roles": roles_data.query(t0, t1)
			}
			

class SpeakerTurnData:
	def __init__(self, path):
		with open(path / "asr_whisper.pkl", "rb") as f:
			data = pickle.load(f)

		self.turns = to_speaker_turns(
			data["output_data"]["speaker_segments"])

		self.path = path

	def iter(self):
		speaker_sentiment = SpeakerSentimentData(self.path)
		speaker_word_class = SpeakerWordClassData(self.path)
		speaker_audio_clf = SpeakerAudioClfData(self.path)
		speaker_segment_clf = SpeakerSegmentClfData(self.path)
		ent_data = EntitiesData(self.path)
		

		for turn in self.turns:
			t0 = float(turn["start"])
			t1 = float(turn["end"])
			emotions = speaker_segment_clf.emotion(t0, t1)
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
				"emotion": top_of_p_list(emotions)
			}


class SimpleCatalog:
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


def encode_video_id(s):
	return base64.b64encode(s.encode("utf8")).decode("ascii")


class Video:
	def __init__(self, path):
		self.path = path

	def export(self, catalog):
		meta = catalog.get(self.path)

		if meta is None:
			return None

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
def tib2nar(pkl, out):
	"""Convert directory of TIB files (PKL) to narrascope JSON data file (OUT)."""
	out = Path(out)
	if not out.suffix == ".json":
		raise click.BadArgumentUsage("output file must end in .json")

	if not out.parent.exists():
		raise click.BadArgumentUsage(f"output path does not exist: {out.parent}")		
	
	try:
		corpus = Corpus.read(Path(pkl))
		corpus.export(out)
	except:
		traceback.print_exc()
	else:
		print(f"successfully exported {out}")


if __name__ == '__main__':
	tib2nar()

