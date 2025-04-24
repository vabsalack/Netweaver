
import netweaver
from netweaver import layers



x_int: int = 4
y_list = list()


def call_me_fuction():
    pass


class CallMeClass:
    pass


print(*globals().items(), sep='\n')
# print(*dir(netweaver), sep='\n')
# print(*netweaver.__dict__.keys(), sep='\n')
# print(netweaver.__package__)