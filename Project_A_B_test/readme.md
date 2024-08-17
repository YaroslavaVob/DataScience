## <center> Анализ результатов А/В-тестирования двух вариантов посадочной страницы </center>
Некая туристическая фирма протестировала два варианта посадочной страницы своего сайта с целью увеличения прибыли за счет более привлекательного дизайна страницы и, как следствие, повышения конверсии и среднего чека.

Нам было поручено провести анализ полученных результатов А/В- тестирования и сделать выводы по метрикам - конверсия и средний чек:\
есть ли разница между контрольной и тестовой группами и можем ли мы говорить, что вариант В статистически значимо отличается от варианта А по заданным метрикам и готов ли вариант В согласно полученным выводам к внедрению или требует доработки.

Наше мини-исследование влючало несколько этапов анализа:
1) подготовка данных, а именно очистка от пропусков и дубликатов (чтобы чистота эксперимента не повлияла на искажение результатов, рекомендовано исключить пользователей, которые увидили оба варианта посадочной страницы);
2) изучение длительности эксперимента и исследование кумулятивных сумм показателей для оценки стабилизации метрик;
3) конечно, визуализация данных и первичная оценка метрик, включая исследование сбалансированности выборок, распределение показателей, изменение метрик во времени;
4) проведение статистических тестов как для конверсии, так и для среднего чека;
5) построение доверительных интервалов для каждой метрики.

По итогам анализа мы смогли увидеть, что одна метрика (конверсия) показала, что оба варианта страницы равнозначны, а вторая метрика (средний чек), что в тестовой группе страница мотивировала покупателей из двух наиболее популярных направлений выбирать более дорогой, остальные направления совпадают с конверсией в контрольной группе. Только этот факт приводил к увеличению среднего чека.\
Так как мы не наблюдали разницы в конверсии в группах, а значит, частота покупок одинакова в группах, и разница в среднем чеке была обнаружена только в двух направлениях туров, то, конечно, утверждать, что вариант В однозначно успешнее и можно его внедрять, мы не можем.



А теперь переходим по ссылке и знакомимся с мини-проектом [здесь](https://github.com/YaroslavaVob/DataScience/blob/main/Project_A_B_test/Project_A_B_test.ipynb).


### Какие библиотеки использовались для анализа:
<font color = 'springblue'>pandas</font>\
<font color = 'springblue'>numpy</font>\
<font color = 'springblue'>seaborn</font>\
<font color = 'springblue'>matplotlib</font>\
<font color = 'springblue'>statsmodel</font>\
<font color = 'springblue'>scipy</font>\


**Проект создан на базе языка Python в Jupyter Notebook.**

Автор: Ярослава Вобшаркян\
(студент школы SkillFactory по курсу Data Science)\
15.08.2024