"""A huggingface dataset for dealing with SNPs."""



import os
import torch

import datasets
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
None (should be bibtex)
"""

_DESCRIPTION = """\
Biobank dataset containing a lot of SNPs
"""

_HOMEPAGE = """\

"""

_URL = "None"


class SNPDataConfig(datasets.BuilderConfig):
    """BuilderConfig for SNPData."""

    def __init__(self, **kwargs):
        """BuilderConfig for SNPData.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class SNPData(datasets.GeneratorBasedBuilder):
    """SNPData v. 0.0.1"""

    BUILDER_CONFIGS = [
        SNPDataConfig(
            name="plain_text",
            version=datasets.Version("0.0.1", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "x": datasets.Value("int8"),
                    "y": datasets.Value("int8"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="None",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""

        filepath =  os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(filepath, "..", "..", "data", "processed", "tensors", "dsmwpred")

        files = os.listdir(path)
        files = sorted(files, key = lambda path: int(path.split("_")[1].split("-")[0]))
        x_files = [f for f in files if f.startswith("x")]
        y_files = [f for f in files if f.startswith("y")]

        for x_path, y_path in zip(x_files, y_files):
            x_path = os.path.join(path, x_path)
            y_path = os.path.join(path, y_path)
            X = torch.load(x_path)
            y = torch.load(y_path)
            for i in range(y.shape[0]):
                X[i]
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph[
                        "context"
                    ]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        answer_starts = [
                            answer["answer_start"] for answer in qa["answers"]
                        ]
                        answers = [answer["text"] for answer in qa["answers"]]
                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "id": qa["id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        key += 1



X[0].unique()
X[10].unique()
y.dtype
for i in y[0]