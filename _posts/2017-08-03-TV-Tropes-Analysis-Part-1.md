---
layout: post
title:  "TV Tropes Analysis Part 1"
date:   2017-08-03 12:55:00 -0400
tags:
- Python
---

This is a proposal of a data project I have been working on: analyze trope usage in movies using data from TV Tropes and IMDB.

<!--more-->

# What Are TV Tropes? Why Study Them?

Tropes are commonly recurring conventions and devices found within various creative works. It is not difficult to recognize a trope when we see it. We can all recognize the Mad Scientist, The Cowboy Cop, or The Chosen One. Overused tropes will probably make us roll our eyes, but the tropes themselves are actually essential to storytelling. Quoting from the famous [TV Tropes website](http://tvtropes.org) that collects pop-culture trope descriptions and examples, "Tropes are not inherently disruptive to a story; however, when the trope itself becomes intrusive, distracting the viewer rather than serving as shorthand, it has become a cliché." Thus, exploring how tropes are used in creative works can help the audience appreciate the various tropes and their function in storytelling, and more importantly, help the creators identify successful and trending plot devices that they can incorporate in their own works.

# The TV Tropes Data Source

The TV Tropes website adopts a wiki style. A typical trope page looks like this:

<img src="/notebooks/TV_Tropes_files/TVTropes_trope_page.png" width="500">

And a typical film page looks like this:

<img src="/notebooks/TV_Tropes_files/TVTropes_film_page.png" width="500">

The data I am interested in collecting is which tropes each movie includes. In other words, I would like to build a dictionary that is something like {movie: [trope1, trope2, ...]}.

Since the TV Tropes website does not encourage web scrapping, I decided to use [DBTropes.org](http://skipforward.opendfki.de/wiki/DBTropes) instead, which is a Linked Data wrapper for TVTropes.org. On their website, they host a downloadable snapshot of [all data in NTriples format](http://dbtropes.org/static/dbtropes.zip), which looks like this:

```text
<http://dbtropes.org/resource/WesternAnimation/TheHobbit> <http://www.w3.org/2000/01/rdf-schema#comment> "The Rankin/Bass adaptation of The Hobbit has an animesque style similar to their adaptation of The Last Unicorn. It might have to do with the fact it was animated by Topcraft, which would later make Nausica\u00E4 of the Valley of the Wind . It also has a lot of Celebrity Voice Actors, including John Huston as Gandalf.Followed up by R-B's adaptation of The Return of the King."@en .
<http://dbtropes.org/resource/WesternAnimation/TheHobbit> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbtropes.org/ont/TVTItem> .
<http://dbtropes.org/resource/WesternAnimation/TheHobbit> <http://skipforward.net/skipforward/resource/seeder/skipinions/hasFeature> <http://dbtropes.org/resource/WesternAnimation/TheHobbit/int_4bb330be> .
<http://dbtropes.org/resource/WesternAnimation/TheHobbit> <http://skipforward.net/skipforward/resource/seeder/skipinions/hasFeature> <http://dbtropes.org/resource/WesternAnimation/TheHobbit/int_9b945553> .
<http://dbtropes.org/resource/WesternAnimation/TheHobbit> <http://www.w3.org/2000/01/rdf-schema#seeAlso> <http://dbtropes.org/resource/Main/Animesque> .
```

# The IMDB Data Source

The IMDB website also prohibits web scrapping. To get information of the movies (eg. genre, release date, director, IMDB ranking, etc.), I used the [ImdbPie library](https://github.com/richardasaurus/imdb-pie), a Python IMDB client using the IMDB json web service made available for their iOS app.

# What Questions Am I After?
Knowing the tropes each movie contains, as well as some basic information about the movies themselves, I can search for the answers to the following questions:

1. Are there any "special" tropes that are seen a lot more / a lot less often in highly rated films?
2. What are the most popular tropes in each film genre? Are there any of the tropes that are associated with good films in certain genres, but not so much for other genres?
3. How has the usage of tropes changed with time? Which tropes are particularly trending now?
4. Can trope profiles be an indicator of film rankings?

# Preliminary Analysis
(The code for this preliminary analysis can be viewed in [this Jupyter Notebook](https://github.com/PeiZhangOSU/TV_trope_analysis/blob/master/TVtropes%20GitHub.ipynb). )

I wrote [this script](https://github.com/PeiZhangOSU/TV_trope_analysis/blob/master/extract_film_data.py) to extract only the data for movies (rather than books, games, etc.) from the DBTropes data set. This subset of data were then parsed using [the rdflib library](https://github.com/RDFLib/rdflib), resulting my desired dictionary that correlates movies with tropes they contain.

For exploratory data analysis, I examined the frequencies of tropes in two groups: (A) the "overall group": tropes that appeared in at least one film in the DBTropes database; (B) the "top rated group": a subgroup of the previous group, consisted of tropes that appeared in at least one film of the IMDB top 250 films. The plot below shows the difference of trope frequencies in each group:

![png](/notebooks/TV_Tropes_files/trope_scatter.png)

In this scatter plot, each dot represent a trope. Each dot's x value is its frequency among all films, while the y value is its frequency among the top rated films. The dots above the diagonal line of y = x (teal colored) are tropes that have higher frequency in the top rated films than in all the films, and the dots below the diagonal line (red colored) are tropes with lower frequency in the top rated films. While most of the dots locate around the diagonal line (meaning these tropes are not "special" -- they are not overrepresented nor underrepresented in the top rated films), those that are farther from the diagonal line could be our candidates for "special" tropes.

Now let's zoom in on the 25 most popular tropes among the IMDB top 250 films:

![png](/notebooks/TV_Tropes_files/most_popular_tropes_top_films.png)

Clearly the pattern of frequency is different in the overall group. The red bars represent the tropes that are seen less often in top films (eg. the most popular trope, "Shout Out"); whereas the teal bars represent the tropes that are more commonly seen in top films (eg. "Oh Crap" -- the top films are probably more intense!).

Altogether, I am confident that the answer to the aforementioned Question 1 is yes -- we do have "special" tropes whose presence/absence are associated with film success. For the other three questions, I plan to perform the following analysis:

- For Question 2: Analyze the frequencies of tropes in each genre.
- For Question 3: Analyze the frequencies of tropes during each decade of releasing date. Fine tune the time period if needed.
(It would be optimal to present the results of Q1-Q3 in an interactive visualization format, like a Tableu visualization or a web page/app, where the user can filter and highlight the film genre / year and tropes.)
- For Question 4: Perform supervised learning with tropes (most likely, selected tropes) as features, IMDB rankings as targets. It is possible that other information of the film could be features too, such as genre and year. I plan to start from random forest algorithm.

All of the results will provide more insight on the domain knowledge about tropes in storytelling. It will benefit the users who want to find inspiration for their creative work, and understand the trend of trope usage.
