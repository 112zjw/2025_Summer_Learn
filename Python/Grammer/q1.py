from statistics import median

list = ["123", 123 , "哈哈哈" ]
print(list)

list.append(2)
print(list)

map = {
    ("张伟", 23):"1234567890",
    ("张伟", 11):"6663339990",
}

map[("张伟" ,66)] = "000000000"


print(("张伟" , 66) in map)

del map[("张伟", 23)]
print(map)

print(map[("张伟" ,66)])

print("12345{0}{1}".format("哈哈","嘿嘿"))

name = "zzz"
age = 99
print(f"ds{name}:{age}")

print(median([5,-19,36]))