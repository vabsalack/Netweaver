1. Type system

    1. Dynamic typing
        1. Python is a dynamically typed language.
            1. Here dynamicallly typed means the variable's type in the program is inferred at runtime and can be reassigned to different types.
            2. PEP 484 introduced type hints, which make it possible to also do static type checking of Python code.
    2. Static typing
        1. Static type checks are performed without running the program. In most statically typed languages, for instance C and Java, this is done as your program is compiled.
    3. Duck typing
        1.  the type or the class of an object is less important than the methods it defines.
        2. Using duck typing you do not check types at all. Instead you check for the presence of a given method or attribute.

2. static type checker is a tool that checks the types of your code without actually running it in the traditional sense.
3. The most common tool for doing type checking is mypy though.

2. pros and cons
    1. pros
        1. Type hints help document your code.
        2. Type hints improve IDEs and linters.
        3. Type hints help you build and maintain a cleaner architecture.
            1. While the dynamic nature of Python is one of its great assets, being conscious about relying on duck typing, overloaded methods, or multiple return types is a good thing.
    2. cons
        Of course, static type checking is not all peaches and cream. There are also some downsides you should consider:
        1. Type hints take developer time and effort to add.
        2. Type hints work best in modern Pythons.
        3. Type hints introduce a slight penalty in startup time
    3. So, should you use static type checking in your own code? 
        1. Well, its not an all-or-nothing question. Luckily, Python supports the concept of gradual typing. 
        2. This means that you can gradually introduce types into your code. Code without type hints will be ignored by the static type checker. 
        3. Therefore, you can start adding types to critical components, and continue as long as it adds value to you.
3. Function Annotations: They are stored in a special .__annotations__ attribute on the function
    1. Annotations were introduced in Python 3.0, originally without any specific purpose. 
    2. Years later, PEP 484 defined how to add type hints to your Python code, based off work that Jukka Lehtosalo had done on his Ph.D. project—mypy.
    3. The main way to add type hints is using annotations. As type checking is becoming more and more common, this also means that annotations should mainly be reserved for type hints.
    4. Sometimes you might be confused by how ypy is interpreting your type hints. For those cases there are special ypy expressions: reveal_type() and reveal_locals().
    5. Even without any annotations mypy has correctly inferred the types.
4. Variable Annotations: Annotations of variables are stored in the module level __annotations__ dictionary
    1. However, sometimes the type checker needs help in figuring out the types of variables as well
    2.  Variable annotations were defined in PEP 526 and introduced in Python 3.6.
    3. You’re allowed to annotate a variable without giving it a value. This adds the annotation to the __annotations__ dictionary, while the variable remains undefined

5. Sequences and Mappings
    1. name: list, version: tuple, options: dict. 
        1. However, this does not really tell the full story
        2. What will be the types of names[2], version[0], and options["centered"]? 
    2. Instead, you should use the special types defined in the typing module. These types add syntax for specifying the types of elements of composite types.
        1. from typing import Dict, List, tuple
        2. name: List[str], version: Tuple[int , int, int], options: Dict[str, bool]
        3. The typing module contains many more composite types, including Counter, Deque, FrozenSet, NamedTuple, and Set
        4. In many cases your functions will expect some kind of sequence, and not really care whether it is a list or a tuple. In these cases you should use typing.Sequence when annotating the function argument
        5. def square(elems: Sequence[float])
        6. Using Sequence is an example of using duck typing. A Sequence is anything that supports len() and .__getitem__(), independent of its actual type.
6. Type aliases
    1. Recall that type annotations are regular Python expressions. That means that you can define your own type aliases by assigning them to new variables.
    2.  
        1. from typing import List, Tuple
        2. Card = Tuple[str, str]
        3. Deck = List[Card]
7. Functions Without Return Values
    1. functions without an explicit return still return None
    2. While such functions technically return something, that return value is not useful. You should add type hints saying as much by using None also as the return type
    3. The annotations help catch the kinds of subtle bugs where you are trying to use a meaningless return value. Mypy will give you a helpful warning
    4. Note that being explicit about a function not returning anything is different from not adding a type hint about the return value
    5. In this latter case, mypy has no information about the return value so it will not generate any warning
    6. As a more exotic case, note that you can also annotate functions that are never expected to return normally. This is done using NoReturn
        1. from typing import NoReturn
        2. def black_hole() -> NoReturn:
            raise Exception("There is no going back ...")

8. The Any type
    1. This means more or less what it says: items is a sequence that can contain items of any type and choose() will return one such item of any type. Unfortunately, this is not that useful.
    2. from typing import Any
    3. While mypy will correctly infer that names is a list of strings, that information is lost after the call to choose() because of the use of the Any type
8. Type variable
    1. A type variable is a special variable that can take on any type, depending on the situation.
    2. 
        from typing import TypeVar
        choosable = TypeVar("choosable")
        or 
        Choosable = TypeVar("Choosable", str, float)
9. Optional type
    1. The Optional type simply says that a variable either has the type specified or is None. 
        def player_order(names: Sequence[str], start: Optional[str] = None) -> Sequence[str]:
    2. An equivalent way of specifying the same would be using the Union type: Union[None, str]
    3. Note that when using either Optional or Union you must take care that the variable has the correct type when you operate on it
10. For class
    1. Type hints for methods 
        1. First of all type hints for methods work much the same as type hints for functions. 
        2. The only difference is that the self argument need not be annotated, as it always will be a class instance.
        3. the .__init__() method always should have None as its return type.
    2. Classes as Types
        1. There is a correspondence between classes and types.
        2. To use classes as types you simply use the name of the class.
        3. Mypy is able to connect your use of Card in the annotation with the definition of the Card class.
        4. This doesn’t work as cleanly though when you need to refer to the class currently being defined. class's method can return a class object. In such case, you can’t simply add -> Deck as the Deck class is not yet fully defined.
        Instead, you are allowed to use string literals in annotations.
        5. @classmethod, cls 


Typing module
1. List, Dict, Tuple and many more composite types
2. List and Tuple are annotated differently. List[int], Tuple[int ,int ,int]
3. from typing import Sequence, sequence[str]
4. function return something, the return value is not useful. -> None
5. function that are never expected to return normally. -> typing.NoReturn
6. Do remember, though, if you use Any the static type checker will effectively not do any type checking.

My mind
1. Annotations in Python   
    1. why not in python
        1. python is dynamically typed. it doesn't bother about varibale types. it throws error whenever the line of execution doesn't find the variable's type respective method or something.
        2. In some IDE, it helps out in suggesting out some associated method in the UI for any object. If the variable is type hinted expilcitly.
    2. how is annotations?
        1. Earlier annotations were introduced for some other reason. In later years, using of type hints were assoicated with annotations. annotations are python expression.
        2. their primary purpose is to provide information to tools and developers.
        3. all variables and composite variables are made basically from primitive types such 
            1. Primitive types typically hold a single immutable value
            2. as integer, float, complex, bool (subclass of int), string, bytes(muttable), bytearray(Immutable) and None types
        4. basic annotations can made by built-in class. int, str, bool, None, list, dict
        5. int, str, list, etc., are both constructors and type hints, but Python distinguishes their use based on context.
        6. If you mistakenly write int() in annotations, Python raises a SyntaxError.
    3.  
        1. Python 3.9+ → Prefer built-in types (list[str], dict[str, int]).
        2. Python 3.8 or earlier → Use typing.List, typing.Dict, etc.
        3. Complex annotations (Unions, Callables, Protocols, etc.) → Always use typing.
    
using of typing module
    1. List, Tuple and Dict
    2. Type, TypeVar
    3. Protocol
    4. Optional
    5. Sequence
    6. union
    7. Callable

using numpy.typing module
1. NDArray - Represents an n-dimensional array of a specific data type.
2. DTypeLike: Denotes valid NumPy data types (e.g., float, int, np.float64).
3. ArrayLike: Represents any object that can be converted to a NumPy array (e.g., lists, tuples)





    



