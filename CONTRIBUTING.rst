Contributing to NeuralHydrology
===============================

Welcome to the NeuralHydrology contribution guide!
--------------------------------------------------

First off, thank you for considering contributing to NeuralHydrology!

The following is a set of guidelines for contributing to NeuralHydrology. They are intended to make contributions as easy as possible for both you as a contributor and us as maintainers of the project.


Kinds of contribution
---------------------

There are many ways to contribute to this project. To name a few:

- File bug reports or feature requests as GitHub issues
- Contribute a bug fix as a pull request
- Contribute a new feature (new model, new dataset, etc.) as a pull request
- Improve the `documentation <https://neuralhydrology.readthedocs.io/>`__ with new tutorials, better descriptions, or even just by fixing typos

The following sections will give some more guidance for each of these options.

Reporting Bugs and feature requests
-----------------------------------
If you've come across some behavior that you think is a bug or a missing feature, the first step is to check if it's already known.
For this, take a look at the `GitHub issues page <https://github.com/neuralhydrology/neuralhydrology/issues>`__.

If you can't find anything related there, you're welcome to create your own issue.

**NOTE**: Please use the `issues page <https://github.com/neuralhydrology/neuralhydrology/issues>`__ for bug reports and feature requests, and leave the `discussions <https://github.com/neuralhydrology/neuralhydrology/discussions>`__ for more open discussions, e.g., project ideas or engaging with other users.


Creating an issue for a bug report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you couldn't find an open issue addressing the problem, open a new one. Please make sure to use a meaningful title and a clear description that includes as much relevant details as possible. Ideally, a bug report should contain:

- A description of how to reproduce the bug (which commands did you execute, did you do any modifications to the code, etc.)
- The full stack trace if any exceptions occurred
- The configuration yaml file you used in your experiments.
- The NeuralHydrology version you used in your experiments (the git commit if you cloned the repo, or the version number in ``neuralhydrology/__about__.py``)
- A description of the dataset you are using in your experiments: Are you using an out-of-the-box CAMELS dataset, or did you create your own csv/netCDF files?
- A list of Python packages you have installed in your environment. If you're using conda, you can create this list via ``conda env export``.

For console output and config files (stack traces, environment files, ...), please try to format them as code directly in the issue (using GitHub's `code block <https://docs.github.com/en/github/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks>`__ syntax) or as a link to a `gist <https://gist.github.com/discover>`__.


Contributing code with a pull request
-------------------------------------
If you wrote some code to fix a bug or provide some new functionality, we use use pull requests (PRs) to merge these changes into the project.

In a PR, please provide a small description of the changes you made, including a reference to the issue that your PR solves (if one exists).
We'll take a look at your PR and do a small code review, where we might outline changes that are necessary before we can merge your PR into the project.
Please don't feel overwhelmed by our change requests---it's entirely normal to have a code review go back and forth a few times before the code is ready to be merged.

Once everyone is happy with the changes in the PR, we'll approve and merge it into the master branch.


Environment setup
~~~~~~~~~~~~~~~~~

The `quick start section <https://neuralhydrology.readthedocs.io/en/latest/usage/quickstart.html>`__ contains information on how to set up the environment (installing the required dependencies and the package itself, as well as downloading data).

To make code changes, you'll need to fork the repository on GitHub.
Describing the process of committing and pushing changes in a git repository goes beyond the scope of this guide.
For more information on how to create pull requests, see for example `this tutorial <https://makeapullrequest.com/>`__ or the `GitHub documentation <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`__.


Code style
~~~~~~~~~~

If possible, we'd be happy if your code is already formatted according to our code style. We use `yapf <https://github.com/google/yapf>`__ with `this configuration file <https://github.com/neuralhydrology/neuralhydrology/blob/master/.style.yapf>`__ to ensure that all our code uses the same line length, indentation style, etc.
If you're using Visual Studio Code as your IDE, you can add the following lines to your settings file (``projectRoot/.vscode/settings.json``):

.. code-block::

    "editor.formatOnSave": true,
    "python.formatting.provider": "yapf",

These settings will make sure that whenever you save changes, yapf will auto-format the changed files according to our style guide.

If you can't get yapf running, don't worry---we can do the formatting as a last step before merging the PR ourselves.


Documentation
~~~~~~~~~~~~~

The `NeuralHydrology documentation <https://neuralhydrology.readthedocs.io/>`__ is automatically generated from the files located in the `docs/` folder.
If your PR introduces any changes that should be documented, you'll find the relevant files there.
Particularly, for new configuration arguments, please add a description of each argument to the `"Configuration Arguments" page <https://github.com/neuralhydrology/neuralhydrology/blob/master/docs/source/usage/config.rst>`__.

To build the documentation webpage locally and see if everything is rendered correctly, simply run ``make html`` in the docs folder. Please check the output of this command to make sure there are no broken links, etc. (these will output red warnings or errors).
Afterwards, you can start a local server via ``python -m http.server`` in ``docs/build/html/``. Once the server is running, you can access the local documentation pages in your browser at the URL ``localhost:8000``.


Continuous Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our GitHub repository contains a suite of test cases that are run automatically on every pull request. These test cases must run successfully before we can merge your contribution. You'll see the status of these test cases at the bottom of your pull request.
Ideally, you've made sure that all tests run smoothly on your local machine before submitting the PR. The test cases use the `pytest <https://docs.pytest.org/>`__ test framework, so you can run all test cases via ``python -m pytest test``, or a specific test case via ``python -m pytest test/test_config_runs.py -k "test_daily_regression_nan_targets"``. If you're using an IDE, it might also auto-detect the test cases for you and provide an option to run them via the IDE's GUI.

Ideally, if you create a PR that fixes a bug or adds some new feature, this PR would also contain one or more test cases that make sure the bug doesn't happen again, or that we don't do changes in the future that would break your new feature. You can find examples for test cases in the ``test/`` folder. If you don't know how to write test cases yourself, don't worry---you can still create a PR and we'll discuss how to create tests during the code review.
