# NLP Issue Labeller
Automatically labels your github issues as ![bug](https://img.shields.io/badge/-bug-f00.svg), ![enhancement](https://img.shields.io/badge/-enhancement-32a4a8.svg) 
![documentation](https://img.shields.io/badge/-Documentation-informational), or skips labelling when unconfident using Natural Language Processing. Finer details:

- Action is triggered whenever an issue is created or editted. 
  
- Action adds (not replace) existing labels of the issue. 

- Disclaimer: auto-labelling takes ~2 minutes to process (as our docker image is pretty large).

> For a demo, simply post an issue in our repository!

## Inputs
`GITHUB_TOKEN`

**Optional** Personal access token of your repository. Uses your access token `${{secrets.GITHUB_TOKEN}}` as default.

`REPOSITORY`

**Optional** Repository to auto-label. Uses current repository `${{github.repository}}` as default.

`CONFIDENCE`

**Optional** Skips prediction when below confidence threshold. 
As model confidence output is in logits, please pick a value from the interval [-10, 4]. Default value is 2.
Note that -10 (or an extremely low value) essentially forces the model to make a prediction for all issues.

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
uses: larrylawl/Auto-Github-Issue-Labeller@main
  with:
    GITHUB_TOKEN: ${{secrets.GITHUB_TOKEN}}
    REPOSITORY: ${{github.repository}} 
    CONFIDENCE: "2"
    FEATURE: "enhancement"
    BUG: "bug"
    DOCS: "documentation"
```

To plug-and-play this action, simply copy `{ROOT}/.github/workflows/main.yaml` from this repository to your own;
remember to keep the same directories to trigger github actions.

## Why you should use our NLP labeller?

**Works out of the box.** No need to come up with regexes (traditional approach). No dependencies added to your codebase.

**Works well.** Our classifier achieved F1 score of 0.8758 and accuracy score of 0.8785 on our test set of repositories (`Flutter`, `OhMyZsh`, and `Electron`).
Well beats the regex-baseline classifier, which achieved F1 score of 0.3634 and accuracy score of 0.5267.

**Only NLP-based auto labeller in the marketplace at time of writing**. Transfer learning with BERT under the hood.

## Citation
This action accompanies our paper [here](TODO). To run the code accompanying our paper, refer to
`paper.md` If you found this action useful, consider citing us at 

```
TODO
```

This work is done in collaboration with Ze Chu (@LiuZechu), Tek In (@OWN463), and Derek (@Derek-Hardy). 
