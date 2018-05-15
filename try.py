from ai import *

print("loading")
load_all()

print("showing loss")
show_loss()

print("testing with teacher forcing")
test_show(force_teacher=True)

print("testing without teacher forcing")
test_show(force_teacher=False)
