age = int(input("Enter your age: "))
ask_nationality = input("Do you want to enter your nationality? (yes/no): ").strip().lower()

if age < 18:
    print("You are a minor." + (" and not eligible for nationality." if ask_nationality == "yes" else ""))
else:
    print("You are eligible to cast a vote." if ask_nationality == "yes" and input("Enter your nationality: ") else "You chose not to enter nationality.")