# tp-datasets
## Pokemon dataset
```python
from tp_datasets import pokemon
img_data, names, types = pokemon.load_data()
```
- Dataset of 256x256 .png images of all Pokemon, their names, and their types
- img_data is already in numpy array format of shape (819,256,256,3).
- Types are already one-hot-encoded (you can get the type order used by pokemon.get_typelist()

## Countries & cities dataset
```python
from tp_datasets import countries_cities as cc
countries = cc.get_country_list()
subcountries = cc.get_subcountry_list()
uk_cities = cc.get_city_list(['United Kingdom'])
all_cities = cc.get_city_list()
```
- ~23000 city-country-subcountry combinations
- Can be used for text generation for example


## Names dataset
```python
from tp_datasets import names
x = names.get_first_names()
```
- 1019 English first names
- 919 from a unisex names dataset, 100 from a most common US name dataset
- Can be used for text generation for example
