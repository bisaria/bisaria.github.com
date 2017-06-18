---
layout: post
title: "Jekyll: Supporting Tagline Error"
date: "June 16, 2017"
category : others
tagline: ""
tags : [jekyll, twitter, tagline]
---
{% include JB/setup %}

In the process of building this websit using Jekyll Boostrap with Twitter theme, I noticed that the tagline that appears next to the page title for posts/pages couldn't be changed even if defined within the YML Front Matter. In fact, the layout templete for the posts and pages had it hard-coded as &lt;small&gt;Supporting tagline&lt;/small&gt;, therefore, one would always see these words.

I just replaced the first 3 lines from page.html and post.html with the [following code](https://github.com/bisaria/bisaria.github.com/blob/master/_posts/2017-06-16-Jekyll-Tagline-Error.md) taken from [here](https://github.com/bendtherules/theme-twitter/blob/37bcce0088296c588324cc7e95e41be32a19fe1d/_includes/themes/twitter/post.html). These page and post templates are located at _includes/themes/twitter.


```
<div class="page-header">
  <h1>{{ page.title }} 
	{% if site.tagline or page.tagline %}
	{% if site.tagline %}
		{% assign _tagline=site.tagline %}
	{% endif %}
	{% if page.tagline %}
		{% assign _tagline=page.tagline %}
	{% endif %}
      <small>{{ _tagline }}</small>
    {% endif %}
</h1>
</div>
```

PS: Still working on stopping the above code snippet from getting set off by kramdown.

