---
layout: page
title: Deep Learning
header: Deep Learning Posts
tagline: ""
group: navigation
---
{% include JB/setup %}

<ul class="posts">
  {% for post in site.posts %}
   {% if post.categories contains 'deeplearning' %}
      <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}  
  {% endfor %} 
</ul>

