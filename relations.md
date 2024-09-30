# Relation Functions in Narrascope

All functions assume the data schema defined in https://github.com/poke1024/narrascope/blob/ec579eb9698f08bf323fa5dba7087b70b7825f7e/schema/schema_2024_07.txt

## actor_relation

`actor_relation(x: Shot, y: Shot, d: Float)` gives a measure of overlap in the actors between shots `x` and `y`, where 1 means full overlap (same actors), and 0 means no overlap (no common actors at all). The value returned is the Jaccard Index between the two sets of actors in the two shots (see [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)). The input parameter `d` controls which faces are regarded as actors based on face size. It ranges from 0 (all faces in a shot are regarded as actors) to 1 (only full-screen faces are regarded as actors). A return value of 0.5 would indicate that faces that are about half the screen height or larger are actors.

Implementation: https://github.com/poke1024/narrascope/blob/ec579eb9698f08bf323fa5dba7087b70b7825f7e/schema/query.txt#L13-L15

## visual_relation

`visual_relation(x: Shot, y: Shot, e: String)` gives a measure of visual similarity of two shots `x` and `y` by computing the embedding `e`. The input parameter `e` selects which image embedding is used for computation and can be any of the following values:

* "siglip"
* "convnextv2"
* "places"
* "kineticsvmae"
* "ssv2vmae"
* "kineticsxclip"

The returned value is a cosine similarity and will thus lie between -1 und 1, where more positive values indicate a higher similarity.

Implementation: https://github.com/poke1024/narrascope/blob/ec579eb9698f08bf323fa5dba7087b70b7825f7e/schema/query.txt#L30-L38

## place_relation

`place_relation(x: Shot, y: Shot, e: String)` is intended to give a measure of place similarity between two shots `x` and `y`. Since we currently do not know how to map this functionality to the underlying features in a way that measures place similarity, `place_relation` is just a different name for `visual_relation`.

Implementation: https://github.com/poke1024/narrascope/blob/ec579eb9698f08bf323fa5dba7087b70b7825f7e/schema/query.txt#L43-L44

## object_relation

`object_relation(x: Shot, y: Shot, e: String)` is intended to give a measure of object similarity between two shots `x` and `y`. Since we currently do not know how to map this functionality to the underlying features in a way that measures object similarity,  `object_relation` is just a different name for `visual_relation`.

Implementation: https://github.com/poke1024/narrascope/blob/ec579eb9698f08bf323fa5dba7087b70b7825f7e/schema/query.txt#L46-L47

## sound_relation

`sound_relation(x: Shot, y: Shot, e: String)` gives a measure of audio similarity of two shots `x` and `y` by computing the embedding `e`. The input parameter `e` selects which image embedding is used for computation and can be any of the following values:

* "wav2vec2"
* "beats"

The returned value is a cosine similarity and will thus lie between -1 und 1, where more positive values indicate a higher similarity.

Implementation: https://github.com/poke1024/narrascope/blob/ec579eb9698f08bf323fa5dba7087b70b7825f7e/schema/query.txt#L54-L62

## region_relation

`region_relation(x: Region, y: Region)` gives a measure of similarity between covered regions. The returned value is a cosine similarity and will thus lie between -1 und 1, where more positive values indicate a higher similarity.

Implementation: https://github.com/poke1024/narrascope/blob/ec579eb9698f08bf323fa5dba7087b70b7825f7e/schema/query.txt#L64-L65

## emotion_relation

`emotion_relation(x: Emotion, y: Emotion, neutral: Float)` gives a measure of similarity between two emotions. The returned value is a cosine similarity and will thus lie between -1 und 1, where more positive values indicate a higher similarity. The input parameter `neutral` defines how much the presense of neutral emotions should degrade the overall score. A value of 5 was used in the example queries as a first estimate for this parameter.

Implementation: https://github.com/poke1024/narrascope/blob/ec579eb9698f08bf323fa5dba7087b70b7825f7e/schema/query.txt#L78-L82

## size_relation_closer

`size_relation_closer(x: Shot, y: Shot, d: Float)` returns `True` if shot `y` is closer than shot `x` (in terms of shot scale), and `False` otherwise. The input parameter `d` controls what is considered `closer` based on the average face size in the two shots. For example, a value of `2` would indicate that for `y` to be consider closer, than `x`, the average face size in `y` would have to be at least twice the average face size of `x `.

Implementation: https://github.com/poke1024/narrascope/blob/ec579eb9698f08bf323fa5dba7087b70b7825f7e/schema/query.txt#L7-L8
