def can_take_medicine():
    age = int(input("Enter the patient's age: "))

    if age >= 18:
        print("Medicine can be given.")
    elif age >= 15:
        weight = float(input("Enter the patient's weight (in kg): "))
        if weight >= 55:
            print("Medicine can be given.")
        else:
            print("Medicine cannot be given.")
    else:
        print("Medicine cannot be given.")

can_take_medicine()
