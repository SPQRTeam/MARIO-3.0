"""
Handles coherence b/w a user-given value and an older value stored on in a file.

Cases accepted without question:
- There is no stored value and the user provided one. Whatever the user gives is taken as good and store it.
- There is a stored value and the user didn't provide anything. Reuse the stored value.
- The user provided the same value that is currently stored. Probably they just repeated the last command on the terminal.

Cases that require explicit confirmation:
- The value given by the user differs from the stored one. The user needs to be made aware they are changing the value.
  If they confirm, the given value overrides the stored one.

Cases that are never accepted:
- There are neither a stored value nor a value given by the user.
  We need to have a value at the end of this, they can't both be null.
  (only applies to "required" fields)
"""
def check_coherence_with_saved_or_update(file_path, user_value, property_name, cli_option, *, required):
    try:
        with open(file_path, "r") as f:
            stored_value = f.read()
    except FileNotFoundError:
        stored_value = None

    if not user_value and not stored_value:  # if field is not given now, it has to have been saved on a previous run
        if required:
            raise FileNotFoundError(f"{file_path.name} not found, you must specify the {property_name} via the {cli_option} option.")
        else:
            return stored_value

    elif user_value and user_value != stored_value:
        if stored_value:  # if field is given now, then it has to be the same as before. If not, the user has to be notified.
            print("!!! WARNING !!!")
            print(f"The {property_name} you just specified is different than the one saved for this game.")
            print("Yours:", user_value)
            print("Saved:", stored_value)
            print(f"Are you SURE you want to override and overwrite the saved {property_name} with yours?")
            print("Type YES to confirm, anything else (or nothing) to abort.")
            if input(" > ").upper() == "YES":
                print(f"Okay. The saved {property_name} will be overwritten.")
            else:
                print("Bye then :)")
                raise ValueError(f"{property_name} specification mismatch. Retry without the {cli_option} option.")
        # update saved, which happens if nothing was saved or if the user chose and confirmed a different field
        with open(file_path, "w") as f:
            f.write(user_value)
        stored_value = user_value

    return stored_value
