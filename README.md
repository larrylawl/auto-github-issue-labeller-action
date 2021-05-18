# Auto Github Issue Labeller Action
![Workflow](https://github.com/larrylawl/Auto-Github-Issue-Labeller/actions/workflows/main.yml/badge.svg)

This action automatically labels your github issues as ![bug](https://img.shields.io/badge/-bug-f00.svg)
, ![enhancement](https://img.shields.io/badge/-enhancement-32a4a8.svg)
![documentation](https://img.shields.io/badge/-documentation-informational), or skips labelling when unconfident using
Natural Language Processing. Some details:

- Action is triggered whenever an issue is created or editted.

- Action only labels unlabelled issue

- **Disclaimer: auto-labelling takes ~2 minutes to process (as our docker image is pretty large).**

For a demo, simply post an issue in our repository!

## Why should you use our NLP labeller?

**Works out of the box.** No need to come up with regexes (traditional approach). No additional dependencies or storage space for models needed.

**Works well.** On our test repositories (`Flutter`, `OhMyZsh`, and `Electron`), our classifier achieved F1 score of
0.8758 and accuracy score of 0.8785. Well beats the regex-baseline classifier, which achieved F1 score of 0.3634 and
accuracy score of 0.5267.

**Only NLP-based auto labeller in the marketplace at time of writing**. Transfer learning with BERT under the hood.

## Inputs

`GITHUB_TOKEN`
**Optional** Personal access token of your repository. Uses your access token `${{secrets.GITHUB_TOKEN}}` as default.

`REPOSITORY`
**Optional** Repository to auto-label. Uses current repository `${{github.repository}}` as default.

`DELTA`
**Optional** Every trigger labels issues up to %DELTA days back. Primary purpose is to label
pre-existing issues on initial setup. Default value is 1.

`CONFIDENCE`
**Optional** Skips prediction when below confidence threshold. As model confidence output is in logits, please pick a
value from the interval [-10, 4]. Default value is 2. Note that -10 (or an extremely low value) essentially forces the
model to make a prediction for all issues.

`FEATURE`
**Optional** Issue label name for `features`-type issues. Default value is `enhancement`.

`BUG`
**Optional** Issue label name for `bug`-type issues. Default value is `bug`.

`DOCS`
**Optional** Issue label name for `docs`-type issues. Default value is `documentation`.

## Outputs

Adds labels to the issue.

## Example usage

```yaml
on:
  issues:
    types: [opened, edited]

jobs:
  auto_label:
    runs-on: ubuntu-latest
    name: Automatic Github Issue Labeller
    steps:
    - name: Label Step
      uses: larrylawl/Auto-Github-Issue-Labeller@main
      with:
        GITHUB_TOKEN: ${{secrets.GITHUB_TOKEN}}
        REPOSITORY: ${{github.repository}}
        DELTA: "1"
        CONFIDENCE: "2"
        FEATURE: "enhancement"
        BUG: "bug"
        DOCS: "documentation"

```

To plug-and-play this action, simply copy `{ROOT}/.github/workflows/main.yaml` from this repository to the same path of your own repository (i.e. `{ROOT}/.github/workflows/main.yaml`). The directory path needs to be respected in order to trigger github actions.

## Credits

This action accompanies our [paper](report.pdf) and [poster](poster.pdf). To run the code accompanying our paper, refer to
`paper.md`

This work is done in collaboration with [Ze Chu](https://github.com/LiuZechu), [Tek In](https://github.com/0WN463),
and [Derek](https://github.com/Derek-Hardy), as part of the module [CS4248](https://knmnyn.github.io/cs4248-2020/)
offered by [A/P Kan Min Yen](https://www.comp.nus.edu.sg/~kanmy/) from
the [National University of Singapore](https://www.comp.nus.edu.sg).

## FAQ

If you ran into this error 

`! [remote rejected] master -> master (refusing to allow an OAuth App to create or update workflow .github/workflows/file.yml without workflow scope)`

Consider updating your Github credentials with a Personal Access Token that enables workflow. Specified by
@eirikvaa's response in this [link](https://stackoverflow.com/questions/64059610/how-to-resolve-refusing-to-allow-an-oauth-app-to-create-or-update-workflow-on#_=_).
