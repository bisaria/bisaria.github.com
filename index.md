---
layout: page
title: Machine Learning Blog
tagline: Machine Learning, Deep Learning
---
{% include JB/setup %}

## Recent Posts

<ul class="posts">
  {% for post in site.posts limit:10 %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>


