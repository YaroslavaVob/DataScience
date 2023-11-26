import pickle  
from datetime import datetime  
from os import path  
  
class Dumper():  
    def __init__(self, archive_dir="archive/"):  
        self.archive_dir = archive_dir  
          
    def dump(self, data):  
        # Библиотека pickle позволяет доставать и класть объекты в файл  
        with open(self.get_file_name(), 'wb') as file:  
            pickle.dump(data, file)  
              
    def load_for_day(self, day):  
        file_name = path.join(self.archive_dir, day + ".pkl")   
        with open(file_name, 'rb') as file:  
            sets = pickle.load(file)  
        return sets  
          
    # возвращает корректное имя для файла   
    def get_file_name(self):   
        today = datetime.now().strftime("%y-%m-%d")   
        return path.join(self.archive_dir, today + ".pkl")