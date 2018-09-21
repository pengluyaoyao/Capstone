from numpy import linspace
import numpy as np
from scipy.stats.kde import gaussian_kde
import dill
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter, Select,Plot
from bokeh.plotting import figure
from bokeh.models.glyphs import Patches
from bokeh.io import curdoc, show
import colorcet as cc
from bokeh.layouts import widgetbox, row, column

Dict = dill.load(open('featurs_dict.pkd', 'rb'))
features = dill.load(open('features.pkd', 'rb'))

#word_count_dict = Dict['total word_count']
#sentence_number_dict = Dict['sentence_number']
#word_features_dict = Dict['word_features']
#count_lemma_dict = Dict['count_lemma']
#spelling_error_dict = Dict['spelling_error']
#count_pos_dict = Dict['count_pos']

# word_count=[]
# Essay_sets=[]
# for key in word_count_dict:
#     n = len(word_count_dict[key])
#     word_count.extend(word_count_dict[key])
#     Essay_sets.extend(np.repeat(key, n))

# df_word_count = pd.DataFrame({'Essay sets': Essay_sets, 'word count': word_count})

def joy(category, data, scale=80):
    return list(zip([category]*len(data), scale*data))

cats = ['essay_set %s' % s for s in np.arange(1,9)]

palette = [cc.rainbow[i*15] for i in range(17)]




feature_dict1 = Dict['total word_count']
feature_dict2 = Dict['sentence_number']
feature_dict3 = Dict['count_lemma']
feature_dict4 = Dict['spelling_error']
xmax1 = (max([max(feature_dict1[n]) for n in feature_dict1]))
xmax2 = (max([max(feature_dict2[n]) for n in feature_dict2]))
xmax3 = (max([max(feature_dict3[n]) for n in feature_dict3]))
xmax4 = (max([max(feature_dict4[n]) for n in feature_dict4]))
x1 = (linspace(0, xmax1, 500))
x2 = (linspace(0, xmax2, 500))
x3 = (linspace(0, xmax3, 500))
x4 = (linspace(0, xmax4, 500))
# Data = {'x': x}
# source = ColumnDataSource(Data)
source1 = ColumnDataSource(data={'x':x1})
source2 = ColumnDataSource(data={'x':x2})
source3 = ColumnDataSource(data={'x':x3})
source4 = ColumnDataSource(data={'x':x4})

p = figure(y_range=cats, plot_width=900, x_range=(0, 900), toolbar_location=None)



for i, cat in enumerate(cats):
    pdf = gaussian_kde(Dict['total word_count'][cat])
    y = joy(cat, pdf(x1))
    source1.add(y, cat)
    r1 = p.patch('x', cat, fill_color=palette[i], fill_alpha=0.8, line_color = None, source=source1)


for i, cat in enumerate(cats):
    pdf = gaussian_kde(Dict['sentence_number'][cat])
    y = joy(cat, pdf(x2))
    source2.add(y, cat)
    r2 = p.patch('x', cat, fill_color=palette[i], fill_alpha=0, line_color = None, source=source2)

for i, cat in enumerate(cats):
    pdf = gaussian_kde(Dict['count_lemma'][cat])
    y = joy(cat, pdf(x3))
    source3.add(y, cat)
    r3 = p.patch('x', cat, fill_color=palette[i], fill_alpha=0, line_color = None, source=source3)

for i, cat in enumerate(cats):
    pdf = gaussian_kde(Dict['spelling_error'][cat])
    y = joy(cat, pdf(x4))
    source4.add(y, cat)
    r4 = p.patch('x', cat, fill_color=palette[i], fill_alpha=0, line_color = None, source=source4)

p.title.text = "The Number of Sentences per Essay Category"
p.title.align = 'center'
p.title.text_font_size = '20pt'
p.title.text_font = 'serif'

    # Axis titles
p.xaxis.axis_label_text_font_size = '14pt'
p.xaxis.axis_label_text_font_style = 'bold'
p.yaxis.axis_label_text_font_size = '14pt'
p.yaxis.axis_label_text_font_style = 'bold'

    # Tick labels
p.xaxis.major_label_text_font_size = '12pt'
p.yaxis.major_label_text_font_size = '12pt'


def update_plot(attr, old, new):

    if 'total word_count'==select1.value:
        r1.glyph.fill_alpha = 0.8
    else:
        r1.glyph.fill_alpha = 0
    if 'spelling_error'==select1.value:
        r4.glyph.fill_alpha = 0.8
    else:
        r4.glyph.fill_alpha = 0
    if 'count_lemma'==select1.value:
        r3.glyph.fill_alpha = 0.8
    else:
        r3.glyph.fill_alpha = 0
    if 'sentence_number'==select1.value:
        r2.glyph.fill_alpha = 0.8
    else:
        r2.glyph.fill_alpha = 0
def update_range(attr, old, new):
    feature_dict = Dict[select1.value]

    xmax = (max([max(feature_dict[n]) for n in feature_dict]))
    p.x_range.end = xmax



select1 = Select(value='total word_count', title='Features',
                 options=['total word_count', 'spelling_error', 'count_lemma', 'sentence_number'])

select1.on_change('value', update_plot)
select1.on_change('value', update_range)

layout = column(widgetbox(select1), p)

curdoc().add_root(layout)