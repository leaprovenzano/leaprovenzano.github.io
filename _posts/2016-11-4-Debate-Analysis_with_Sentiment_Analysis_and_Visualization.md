---
layout: post
title: Visualization of Presidential Debate Language Using NLTK and Andreas Mueller's Super Awesome Wordcloud Library.
---


![Clinton]({{ site.baseurl }}/images/clinton.jpg "Clinton")Clinton

![Trump]({{ site.baseurl }}/images/trump.jpg "Trump")Trump

I have been messing around with the transcripts from the recent US presidential debates. I nabbed them off politico, though there's also a [Kaggle](https://www.kaggle.com/mrisdal/2016-us-presidential-debates). I was able to generate some wordclouds using [Andreas Mueller's excellent worldcloud library](https://github.com/amueller/word_cloud). By default wordcloud has a random color function but you can easily pass it your own color function to the wordcloud constructor as well. I thought it would be interesting to do some sentiment analysis and use the positive and negative scores to color the wordclouds (blue being positive and red being negative, you could say I'm a little biased). 

```python
    def gen_cloud(data):
        counts = [(w, data[w]['count']) for w in data]
        def sent_color_function(word=None, font_size=None, position=None,
                                orientation=None, font_path=None, random_state=None):

            r, g, b = 126 + int(255 * data[word]['neg']), 126, 126 + int(255 * data[word]['pos'])
            if r > 255:
                v = r - 255
                b = max(b - v, 0)
                g = max(g - v, 0)
                r = 255
            if b > 255:
                v = b - 255
                r = max(r - v, 0)
                g = max(g - v, 0) 
                b = 255
            return "rgb({}, {}, {})".format(r, g, b)

        wordcloud = WordCloud(  max_font_size = 100,
                                width= 800, 
                                height = 400,
                                color_func=sent_color_function).generate_from_frequencies(counts)
        return wordcloud
```

This was my first experiement with sentiment analysis. After some basic data cleaning I broke tokenized each candidate's text into sentences and used the [VADER](http://www.nltk.org/_modules/nltk/sentiment/vader.html) SentimentIntensityAnalyzer to gather scores for each sentence. Then After POS tagging and lemmatization, I used NLTK sentiwordnet synsets to get and average of all positive and negative scores from synsets given individual words given their part of speech, memoizing as I went along. It's worth noting that sentiwordnet only handles verbs, adverbs, nouns, and adjectives - but this was sufficient for my purposes.

```python
    def word_senti_score(word, POS):
        """returns nltk sentiwordnet...
        Args:
            word (str): Description
            POS (str): part of speech should be 
                       must be in NLTK sentiwordnet
        Returns:
            TYPE: pos & neg values... skips neu
        """
        p, n = 0., 0.
        try:
            p, n =  WORD_SCORES[(word, POS)]
        except KeyError:
            scores = list(sentiwordnet.senti_synsets(word, POS))
            if scores: # this will average all synset words for given POS
                p = sum([s.pos_score() for s in scores])/ len(scores)
                n = sum([s.neg_score() for s in scores])/len(scores)
            WORD_SCORES[(word, POS)] = (p, n)
        return p, n
```

I combined the positive and negative word scores for each use with the scores from that word's originating sentence and took the mean of all usage values to be that unique word's score. So that if a word is generally neutral or generally positive but is used repeatedly in negative sentences this will be reflected in the unique score for that word. In this way each candidates positivity and negativity given a certain word will vary, as the scores are unique to their usage. 

Here is a [link to the full repository on github](https://github.com/leaprovenzano/debate_language_analysis) if you're interested in checking it out. There's also an [ipython notebook](https://github.com/leaprovenzano/debate_language_analysis/blob/master/notebook_debates.ipynb) in there with a bit more explanation of my process





----------

* Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

