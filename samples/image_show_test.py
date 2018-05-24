import pyvision as pv

im = pv.Image(pv.LENA)
out = im.show('lena',delay=30)

im = pv.Image(pv.BABOON)
out = im.show('baboon',delay=3000)

im = pv.Image(pv.AIRPLANE)
out = im.show('baboon',delay=3000)

print("im.show returned:",out)