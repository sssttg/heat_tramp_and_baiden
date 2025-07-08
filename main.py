# модуль для обработки данных
import pandas as pd
# модули для работы с тепловыми картами
import seaborn as sns
import matplotlib.pyplot as plt

# путь к файлу
data_url = 'data/biden_trump_tweets.csv'
# читаем
df = pd.read_csv(
    data_url,
    parse_dates=['date_utc'],
    dtype={
        'hour_utc': int,
        'minute_utc': int,
        'id': str
        }
)
# print(df.head())

# объединяем события, которые произошли у каждого юзера в одно и то же время
g = df.groupby(['hour_utc', 'minute_utc', 'username'])

# кол-во твитов в каждой группе
tweet_cnt = g.id.nunique()
# print(tweet_cnt.head())

# начинаем обработку данных с Джо Байдена
# получаем доступ к группе строк и столбцов,
# чтобы сформировать новый датасет для тепловой карты
jb_tweet_cnt = tweet_cnt.loc[:, :, 'JoeBiden'].reset_index().pivot(
    index='hour_utc',
    columns='minute_utc',
    values='id',
    )
# меняем NaN на 0, чтобы у нас в данных остались только численные значения
jb_tweet_cnt.fillna(0, inplace=True)
# print(jb_tweet_cnt.iloc[:10, :9])

# добавляем отсутствующие часы
jb_tweet_cnt = jb_tweet_cnt.reindex(range(0, 24), axis=0, fill_value=0)
# то же самое с минутами
jb_tweet_cnt = jb_tweet_cnt.reindex(
    range(0, 60), axis=1, fill_value=0
    ).astype(int)
# print(jb_tweet_cnt.iloc[:20, :18])

# то же самое для трампа
dt_tweet_cnt = tweet_cnt.loc[:, :, 'realDonaldTrump'].reset_index().pivot(
    index='hour_utc', columns='minute_utc', values='id'
)
dt_tweet_cnt.fillna(0, inplace=True)
dt_tweet_cnt = dt_tweet_cnt.reindex(range(0, 24), axis=0, fill_value=0)
dt_tweet_cnt = dt_tweet_cnt.reindex(
    range(0, 60), axis=1, fill_value=0
    ).astype(int)
# print(dt_tweet_cnt.iloc[:20, :18])

# готовимся показать несколько графиков в одном фрейме
fig, ax = plt.subplots(2, 1, figsize=(24, 12))

#  перебираем оба датасета по одному элементу
for i, d in enumerate([jb_tweet_cnt, dt_tweet_cnt]):
    # получаем значения минут и часов для разметки осей
    labels = d.applymap(lambda v: str(v) if v == d.values.max() else '')
    # делаем тепловую карту
    sns.heatmap(
        d,
        cmap='viridis',  # тема
        annot=labels,  # разметка осей - labels
        annot_kws={'fontsize': 11},  # размер шрифта
        fmt='',  # говорим, что с метками работаем как со строками
        square=True,  # квадратные ячейки
        vmax=40,  # максимум
        vmin=0,  # минимум
        linewidth=0.01,  # разлиновка сеткой
        linecolor='#222',  # цвет сетки
        ax=ax[i],  # значение каждой клетки берем в соответствии с датасетом
    )
# подписываем оси
ax[0].set_title('@JoeBiden')
ax[1].set_title('@realDonaldTrump')
ax[0].set_ylabel('Распределение по часам')
ax[1].set_ylabel('Распределение по часам')
ax[0].set_xlabel('')
ax[1].set_xlabel('Распределение по минутам')

# сохранеям результат
plt.tight_layout()
plt.savefig('final.png', dpi=120)
plt.show()
