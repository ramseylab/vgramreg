def change_letter_size(function):
    def wrapper(name):
        print("runs before calling function size changer")
        name = name.lower()
        function(name)
        print("runs after calling function size changer")
        return 
    
    return wrapper

def split_letters(function):
    def wrapper(name):
        print("runs before calling function splitter")
        name = name.split(' ')
        function(name)
        print("runs after calling function splitter")
        return 
    
    return wrapper


@change_letter_size
@split_letters
def name_calling(name):
    print("This is name calling")
    print(name)

name_calling("SAN MAN")