import matplotlib.colors as mc
import string

def name_to_rust(name: str) -> str:
    name = name.split(":")[-1].strip()
    name = name.replace("_", " ")
    name = name.replace("/", " Or ")
    name = name.replace("'", "")
    name = string.capwords(name)
    name = name.replace(" ", "")
    return name

def colors_to_rust(colors):
    names = sorted(colors)
    for name in names:
        print("    " + name_to_rust(name) + ",")
    print()
    for name in names:
        (r, g, b) = mc.to_rgb(name)
        print("            " + name_to_rust(name),
              f"=> [{r}, {g}, {b}, 1.],")

colors_to_rust(mc.CSS4_COLORS)

colors_to_rust(mc.XKCD_COLORS)
