from zxcvbn import zxcvbn

print("🔐 zxcvbn Password Tester (type 'exit' to quit)\n")

while True:
    password = input("Enter password: ")

    if password.lower() == "exit":
        break

    result = zxcvbn(password)

    score = result["score"]  # 0–4
    feedback = result["feedback"]

    print("\n--- Results ---")
    print(f"Score (0-4): {score}")

    # simple interpretation
    levels = ["Very Weak", "Weak", "Fair", "Good", "Strong"]
    print(f"Strength: {levels[score]}")

    if feedback["warning"]:
        print(f"⚠️ Warning: {feedback['warning']}")

    if feedback["suggestions"]:
        print("💡 Suggestions:")
        for s in feedback["suggestions"]:
            print(f" - {s}")

    print("----------------\n")