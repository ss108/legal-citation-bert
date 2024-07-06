# Legal Citation BERT
Code for https://huggingface.co/ss108/legal-citation-bert

Legal Citation BERT is a version of `dslim/bert-base-NER` trained to recognize
American legal citations. Currently, it handles standard Bluebook caselaw and
statute citations (including rules of procedure) quite well.

## Goal

The basic purpose of this model is to help academics and legal tech builders
identify a wide range of legal citations cost-effectively at-scale. 

If your needs in this area are modest, I would recommend looking to an LLM in
the first instance. I can guarantee you that, at the time of this writing in
June 2024, GPT-4 can do a better job than this library. It knows the Bluebook
well, and it also understands California's citation style.

However, if you're sitting on troves of legal documents with citations, with more documents
coming in daily--e.g. if you are the Free Law Project
(https://www.courtlistener.com/) or Trellis (trellis.law) (my employer at the
time of this writing), then going using a standard LLM API, or even a
"self-hosted" commerical LLM on Azure or Bedrock, isn't going to cut it.

You can try seeing if an open-source LLM can do the job, but ultimately, a model like
this is cheaper and hopefully there is some benefit in its being specifically
tailored for the task.

(And if nothing else, it's good for my resume/career)

## Example Output; Usage

The plan is that one wouldn't use this repo/model directly unless one wished to
augment its training. Instead, I am going to write a library that invokes this
model from a user-configured source (i.e. Beam | Modal | HuggingFace |
SageMaker | on-device/local). 

That library will take the raw model output, parse it, and return back to the
user nice little packaged Python objects. A very rudimentary example of what I
mean can be found in `src/benchmarking/temp_aggregation.py` 

Additionally, the library will be able to augment the model predictions with data from external sources, such as the Courtlistener API.

That library should be going up soon.

Anyways, here are the labels the model uses (literally copied from the training code):

```
CASE_LABELS = [
    "B-CASE_NAME",
    "I-CASE_NAME",
    "B-VOLUME",
    "I-VOLUME",
    "B-REPORTER",
    "I-REPORTER",
    "B-PAGE",
    "I-PAGE",
    "B-PIN",
    "I-PIN",
    "B-COURT",
    "I-COURT",
    "B-YEAR",
    "I-YEAR",
]

STAT_LABELS = [
    "B-TITLE",
    "I-TITLE",
    "B-CODE",
    "I-CODE",
    "B-SECTION",
    "I-SECTION",
]

SHORT_LABELS = ["B-ID", "I-ID", "B-SUPRA", "I-SUPRA"]


ALL_LABELS = CASE_LABELS + STAT_LABELS + SHORT_LABELS + ["O"]
```

Here are a couple poorly formatted examples:

(Note that ofc the NER model doesn't return token-label pairs, it just returns
an array of labels. The construction of the pairs is for convenience, though I'm
not sure it actually ends up being that readable.)

EXAMPLE TEXT: Hi, I am a simple example Hanover Star Milling Co. v. Metcalf, 240
U.S. 403, 412 (1916).

Predicted Labels for each token: 
```
[[CLS]: O, Hi: O, ,: O, I: O, am: O, a: O, simple: O, example: O, Hanover: B-CASE_NAME, Star: I-CASE_NAME, Mill: I-CASE_NAME, ##ing: I-CASE_NAME, Co: I-CASE_NAME, .: I-CASE_NAME, v: I-CASE_NAME, .: I-CASE_NAME, Met: I-CASE_NAME, ##cal: I-CASE_NAME, ##f: I-CASE_NAME, ,: O, 240: B-VOLUME, U: B-REPORTER, .: I-REPORTER, S: I-REPORTER, .: I-REPORTER, 40: B-PAGE, ##3: I-PAGE, ,: O, 41: B-PIN, ##2: I-PIN, (: O, 1916: B-YEAR, ): O, .: O, [SEP]: O]
```

Note that basically the only puncutation that receives a label is that which is
part of the case name. The puncutation that occurs as part of "citation
formatting", such as the comma in between the starting page and the pin cite,
are not labeled--they are not part of the substance of the citation, rather they are
instead part of the grammar by which the actual information of the citation is
communicated.

EXAMPLE TEXT: Laws exist 21 U.S.C. § 79

Predicted Labels:
```
[[CLS]: O, Laws: O, exist: O, 21: B-TITLE, U: B-CODE, .: I-CODE, S: I-CODE, .: I-CODE, C: I-CODE, .: I-CODE, §: O, 79: B-SECTION, [SEP]: O]
```

Note that the section symbol itself is not included on the same logic as above.

EXAMPLE TEXT: May I interest you in a short citation--Fux, 76 F.3d at 89.

Predicted Labels:
```
[[CLS]: O, May: O, I: O, interest: O, you: O, in: O, a: O, short: O, citation: O, -: O, -: O, Fu: B-CASE_NAME, ##x: I-CASE_NAME, ,: O, 76: B-VOLUME, F: B-REPORTER, .: I-REPORTER, 3: I-REPORTER, ##d: I-REPORTER, at: O, 89: B-PIN, .: O, [SEP]: O]
```

## Limits

Current known weaknesses: 

*  Doesn't properly handle citations to various forms of unpublished or
   unofficial opinions (specifically, it often wrongly labels the reporter year
   of a WestLaw or Lexis citation as all or partially representing the reporter
   volume; e.g. in something like "2022 WL 1363209", it usually mischaracterizes
   the last "2" in "2022" as "I-VOLUME")
*  Doesn't properly label notes as part of pin cites.
*  Thinks legislative materials are statutes.
*  Sometimes, due to OCR, document page numbers are interjected into the middle
   of citations, and this messes things up.

The first three are all solvable via more training data showcasing the correct labeling.


## Training Data
Some of the training text was generated by LLMs, but the substantial majority
consists of real-life Federal court filings from Courtlistener, which I got from Pile-of-Law's
dataset(https://huggingface.co/datasets/pile-of-law/pile-of-law). 

The subset of the Pile-of-Law CL Docket Entries dataset that was used consists
of approximately the first 50 documents. Since the entire corpus of raw data is
currently small enough to fit in a GitHub repo, precisely which Courtlistener documents were
used can be gleaned from the file names in `raw_data`

Credits:
@misc{hendersonkrass2022pileoflaw,
  url = {https://arxiv.org/abs/2207.00220},
  author = {Henderson*, Peter and Krass*, Mark S. and Zheng, Lucia and Guha, Neel and Manning, Christopher D. and Jurafsky, Dan and Ho, Daniel E.},
  title = {Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset},
  publisher = {arXiv},
  year = {2022}
}

