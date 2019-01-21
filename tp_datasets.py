import os, cv2, csv
import numpy as np

def __csv_to_2D_list(path):
    '''
    Helper method for reading one column from a CSV file
    Parameters
    ----------
    path: string
        Absolute path to the CSV file
    Return
    ----------
    2D list representing the whole CSV file
    Format: [row1,row2,row3...], where rows are also lists
    '''
    with open(path) as f:
        table = f.readlines()
    for i, row in enumerate(table):
        temp = [entry.replace("\"","") for entry in row.strip().split(",")]      
        table[i] = temp
    return table

def __csv_col_to_array(path,column,skip_first=True,to_lowercase=False):
    '''
    Helper method for reading one column from a CSV file
    Parameters
    ----------
    path: string
        Absolute path to the CSV file
    column: int
        Index of column to read, starting with 0
    skip_first (optional): boolean
        Whether to skip the first row (usually used for headers)
        Default is True
    to_lowercase (optional): boolean
        Whether to turn the words to lowercase
        Default is False
    Return
    ----------
    List of entries in the specified column
    '''
    table = __csv_to_2D_list(path)
    entries = []
    for _, row in enumerate(table):
        if to_lowercase and type(row[column]) is str:
            entries.append(row[column].lower())
        elif not to_lowercase:
            entries.append(row[column])
    if skip_first:
        return entries[1:]
    else:
        return entries

def __get_abs_data_path(dir_name,file_name):
    '''
    Helper method for accessing various files in the 'Datasets' directory
    Parameters
    ----------
    dir_name: string
        Name of the directory the CSV file is in
    file_name: string
        Name of the CSV file
    Return
    ----------
    The absolute path to the specified CSV file
    '''
    current_dir = os.path.dirname(os.path.realpath(__file__))
    datasets_dir = os.path.join(current_dir,"Datasets")
    path = os.path.join(datasets_dir,dir_name)
    path = os.path.join(path,file_name)
    return path


class names:
    
    @staticmethod
    def get_first_names(to_lowercase=False):
        '''
        Returns 1019 English first names
        919 from a unisex names dataset, 100 from a common US names dataset
        Unisex names dataset source: https://github.com/fivethirtyeight/data/unisex-names
        Common US names dataset source: https://github.com/fivethirtyeight/data/most-common-name
        Parameters
        ----------
        to_lowercase (optional): boolean
            Whether to turn the words to lowercase
            Default is False
        Return
        ----------
        List of first names
        '''
        path = __get_abs_data_path(dir_name="unisex-names",file_name="unisex_names_table.csv")
        names = __csv_col_to_array(path,1,skip_first=True,to_lowercase=to_lowercase)

        path = __get_abs_data_path(dir_name="most-common-name",file_name="new-top-firstNames.csv")
        names += __csv_col_to_array(path,1,skip_first=True,to_lowercase=to_lowercase)

        return names

class countries_cities:

    @staticmethod
    def get_country_list(to_lowercase=False):
        '''
        Dataset source: https://github.com/fivethirtyeight/data/world-cities
        Parameters
        ----------
        to_lowercase (optional): boolean
            Whether to turn the words to lowercase
            Default is False
        Return
        ----------
        List of country names, sorted alphabetically
        '''
        path = __get_abs_data_path(dir_name="World_Cities",file_name="world-cities.csv")
        countries = set(__csv_col_to_array(path,1,skip_first=True,to_lowercase=to_lowercase))
        return sorted(list(countries))
    
    @staticmethod
    def get_city_list(from_countries=None):
        '''
        Dataset source: https://github.com/fivethirtyeight/data/world-cities
        Parameters
        ----------
        from_countries (optional): list of strings (country names)
            List of countries from which to return city names.
            Leave unspecified to get all city names
        Return
        ----------
        List of city names
        '''
        cities = []
        path = __get_abs_data_path(dir_name="World_Cities",file_name="world-cities.csv")
        table = __csv_to_2D_list(path)
        for _, row in enumerate(table):
            if from_countries is None or row[1] in from_countries:
                cities.append(row[0])
                     
        return cities
    
    @staticmethod
    def get_subcountry_list(from_countries=None):
        '''
        Dataset source: https://github.com/fivethirtyeight/data/world-cities
        Parameters
        ----------
        from_countries : list of strings (country names)
            List of countries from which to return subcountries.
            Leave unspecified to get all subcountry names
        Return
        ----------
        List of subcountry names
        '''
        subcountries = []
        path = __get_abs_data_path(dir_name="World_Cities",file_name="world-cities.csv")
        table = __csv_to_2D_list(path)
        for _, row in enumerate(table):
            if from_countries is None or row[1] in from_countries:
                subcountries.append(row[2])
                      
        return subcountries

class pokemon:

    @staticmethod
    def load_data(include_alpha=False, return_full_names=True):
        '''
        Images source: https://www.kaggle.com/kvpratama/pokemon-images-dataset
        Manually added names + types
        Parameters
        ----------
        include_alpha (optional): boolean
            Whether to include the alpha channel
            Default is False
        return_full_names (optional): boolean
            Whether to return full names (Mega Charizard, Rotom Wash),
            or just the main name (Charizard, Rotom)
            Default is True
        Return
        ----------
        A tuple (image array, names, types)
        Image array : Numpy array of shape (819,256,256,channels (3 or 4 depending on include_alpha))
        Names : List of Pokemon names
        Types : Numpy array of shape (819,18), contains one-hot-encoded types
        To retrieve the order of types used in one-hot-encoding, call pokemon.get_typelist()
        '''
        current_dir = os.path.dirname(os.path.realpath(__file__))
        datasets_dir = os.path.join(current_dir,"Datasets")
        pokemon_dir = os.path.join(datasets_dir,"pokemon")
        img_dir = os.path.join(pokemon_dir,"images")
        filenames = os.listdir(img_dir)

        if include_alpha:
            channels = 4
            imread_mode = cv2.IMREAD_UNCHANGED
        else:
            channels = 3
            imread_mode = cv2.IMREAD_COLOR
        

        numpy_arrays = np.zeros(shape=(len(filenames),256,256,channels),dtype='uint8')
        names, types = [], np.zeros(shape=(819,18))

        with open(os.path.join(pokemon_dir,"poke_info.csv")) as f:
            reader = csv.reader(f, delimiter=',')
            i = 0
            for row in reader:
                if i == 0:
                    # Skipping the first row, it's only headers
                    i = 1
                    continue
                abs_path = os.path.join(img_dir,row[0])
                numpy_arrays[i-1] = cv2.imread(abs_path,imread_mode)
                names.append(row[1])
                types[i-1] = np.array(row[2:])
                i+=1
        if return_full_names:
            return numpy_arrays, names, types
        else:
            for i, name in enumerate(names):
                words = name.split(' ')
                if name == 'Mr. Mime': # The only two word 'main name'
                    pass
                elif words[0] == 'Mega': # Other 'extra words' are second
                    names[i] = words[1]
                else:
                    names[i] = words[0]
            return numpy_arrays, names, types

    @staticmethod
    def get_typelist():
        '''
        Return
        ----------
        List of Pokemon types used in the .csv file and in the one-hot-encoded array you get with pokemon.load_data()
        '''
        return ["Normal","Fighting","Flying","Poison","Ground","Rock","Bug","Ghost","Steel",
         "Fire","Water","Grass","Electric","Psychic","Ice","Dragon","Dark","Fairy"]

