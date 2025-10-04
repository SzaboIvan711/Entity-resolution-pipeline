import re
import random
from typing import List
import numpy as np
import pandas as pd
from faker import Faker

def gen_customers(
    n_unique: int = 500,
    dup_rate: float = 0.30,
    max_dups_per: int = 2,
    locale: str = "en_US",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic customer dataset with optional duplicated/perturbed records.
    """

    rng = random.Random(seed)
    np.random.seed(seed)
    fake = Faker(locale)
    Faker.seed(seed)

    def only_digits(s: str, n: int = 5) -> str:
        """Keep only digits and pad/truncate to n characters."""
        d = "".join(ch for ch in s if ch.isdigit())
        return (d + "0"*n)[:n] if n else d

    def noisy_str(s: str) -> str:
        """Inject light noise: delete/swap/replace characters."""
        if not s or len(s) < 2:
            return s
        s_list = list(s)
        # character replacement
        if rng.random() < 0.35:
            i = rng.randrange(len(s_list))
            if s_list[i].isalpha():
                s_list[i] = chr(((ord(s_list[i].lower()) - 97 + 1) % 26) + 97)
        # character deletion
        if rng.random() < 0.25 and len(s_list) > 1:
            i = rng.randrange(len(s_list))
            s_list.pop(i)
        # swap adjacent characters
        if rng.random() < 0.25 and len(s_list) > 2:
            i = rng.randrange(len(s_list) - 1)
            s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
        return "".join(s_list)

    def tweak_email(email: str) -> str:
        """Introduce mild typos in email (local part or TLD)."""
        if rng.random() < 0.6:
            # small typo in the local part
            local, domain = email.split("@")
            if len(local) > 3:
                i = rng.randrange(len(local))
                local = local[:i] + "" + local[i+1:]
            email = f"{local}@{domain}"
        if rng.random() < 0.3:
            email = email.replace(".com", ".co", 1)
        return email

    def tweak_phone(phone: str) -> str:
        """Randomly replace 1–2 digits in the phone number."""
        phone = list(phone)
        for _ in range(rng.randint(0, 2)):
            i = rng.randrange(len(phone))
            if phone[i].isdigit():
                phone[i] = str(rng.randrange(10))
        return "".join(phone)

    rows: List[dict] = []
    uid = 0
    for _ in range(n_unique):
        uid += 1
        name = fake.name()
        street = f"{fake.street_name()} {rng.randint(1, 199)}"
        city = fake.city()
        zipc = only_digits(fake.postcode(), 5)
        email_local = re.sub(r"[^a-z0-9]+", ".", name.lower())
        email = f"{email_local}@example.com"
        phone = "0" + "".join(str(rng.randrange(10)) for _ in range(9))

        base = dict(uid=uid, name=name, street=street, city=city, zip=zipc, email=email, phone=phone)
        rows.append(base)

        # Create duplicates with probability dup_rate
        if rng.random() < dup_rate:
            for _ in range(rng.randint(1, max_dups_per)):
                rows.append(dict(
                    uid=uid,
                    name=noisy_str(name),
                    street=noisy_str(street),
                    city=city if rng.random() < 0.9 else noisy_str(city),
                    zip=zipc,  # ZIP often matches
                    email=email if rng.random() < 0.7 else tweak_email(email),
                    phone=phone if rng.random() < 0.7 else tweak_phone(phone),
                ))

    df = pd.DataFrame(rows).reset_index(drop=True)
    # Helpful service fields
    df.insert(0, "row_id", np.arange(1, len(df) + 1))
    return df

# Example usage:
if __name__ == "__main__":
    df = gen_customers(n_unique=500, dup_rate=0.3, max_dups_per=2, locale="en_US", seed=42)
    print(df.head(10))
    print("\nTotal rows:", len(df), " | Unique customers (uid):", df["uid"].nunique())
    # Save to CSV
    df.to_csv("data/customers_synthetic.csv", index=False)
    print('\nFile saved: customers_synthetic.csv')
