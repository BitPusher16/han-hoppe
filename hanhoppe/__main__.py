# use this if you are running with $ python hanhoppe
#import tile_utils as tu

# use this if you are running with python -m hanhoppe
import hanhoppe.tile_utils as tu
import hanhoppe.command_line as cl

tu.say_hello()
cl.parse_args()

print('hey')

# __name__ will be __main__ if called with either of:
# $ python hanhoppe
# or
# $ python -m hanhoppe
if __name__ == '__main__':
    print('name is __main__')
else:
    print('name is not __main__')
