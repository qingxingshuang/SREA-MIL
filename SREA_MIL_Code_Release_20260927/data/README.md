# Data interface

The private cervical cytology features and patient/slide identifiers are not
distributed. Prepare one NumPy file per available stream and slide. Each file
must be a float array of shape N by 1280.

The CSV manifest requires four columns:

- slide_id: a unique, de-identified slide key;
- label: 0/NILM/normal or 1/abnormal;
- abnormal_feature_path: suspected-abnormal candidate features;
- normal_feature_path: normal-candidate features.

Paths may be absolute or relative to the manifest. One stream may be empty,
but every slide must contain at least one valid stream. The loader samples at
most 512 instances from each stream during training using a deterministic key
derived from seed, slide ID, stream, and epoch. Validation and held-out loading
use a fixed epoch key.

The split JSON contains development, held_out, and exactly five inner_folds.
The real split file is intentionally omitted because it contains private
dataset identifiers.
