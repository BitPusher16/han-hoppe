# use this if you are running with $ python han_hoppe
#import tile_utils as tu

# use this if you are running with $ python -m han_hoppe
import han_hoppe.tile_utils as tu

import han_hoppe.command_line as cl
import sys

#print(tu.qk_to_x_y_lod("021230021313"))
#print(tu.x_y_lod_to_qk(655,1429,12))
#sys.exit()

cl.parse_args()

# __name__ will be __main__ if called with either of:
# $ python hanhoppe
# or
# $ python -m hanhoppe
# but __name__ will not be __main__ if hanhoppe is imported.
if __name__ == '__main__':
    print('name is __main__')
else:
    print('name is not __main__')
