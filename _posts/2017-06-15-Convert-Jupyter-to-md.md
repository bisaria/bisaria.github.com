---
layout: post
title: "Jupyter: Convert to Markdown"
date: "June 16, 2017"
category : others
tagline: ""
tags : [jupyter, markdown]
---
{% include JB/setup %}

Converting Jupyter notebook to markdown format:

```
$ ipython nbconvert jekyll_test.ipynb --to markdown
```

Although it does through the following warning,
```
[TerminalIPythonApp] WARNING | Subcommand `ipython nbconvert` is deprecated and
will be removed in future versions.
[TerminalIPythonApp] WARNING | You likely want to use `jupyter nbconvert` in the
 future
```
but using `jupyter nbconvert` for some reason did not work for me.
 
Jupyter notebook can be convert into various other file formats like pdf, html etc. as shown [here](https://ipython.org/ipython-doc/3/notebook/nbconvert.html). Latest documentation with the new command `jupyter nbconvert` can be found [here](http://nbconvert.readthedocs.io/en/latest/usage.html#convert-markdown).

