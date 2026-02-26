#!/usr/bin/env python3
# pro.py – Comprehensive Time Value of Money Solver

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Try to import scipy for robust root finding
# ----------------------------------------------------------------------
try:
    from scipy.optimize import newton, brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ----------------------------------------------------------------------
# Core TVM functions
# ----------------------------------------------------------------------

def tvm_fv(pv=None, pmt=0, n=1, rate=0, due=False, compound_per_year=1):
    """
    Compute future value.
    Handles lump sum (pmt=0) and annuities.
    If compound_per_year == 0, simple interest is used (only lump sum, pmt ignored).
    """
    if compound_per_year == 0:
        # Simple interest (lump sum only)
        if pmt != 0:
            raise ValueError("Simple interest does not support payments.")
        return pv * (1 + rate * n)

    # Convert to per‑period rate and number of periods
    r = rate / compound_per_year
    n_periods = n * compound_per_year

    if pv is None:
        pv = 0
    # Future value of lump sum
    fv_lump = pv * (1 + r) ** n_periods

    if pmt == 0:
        return fv_lump

    # Future value of annuity
    if due:
        # Annuity due: payments at beginning
        fv_annuity = pmt * (1 + r) * ((1 + r) ** n_periods - 1) / r
    else:
        # Ordinary annuity: payments at end
        fv_annuity = pmt * ((1 + r) ** n_periods - 1) / r

    return fv_lump + fv_annuity


def tvm_pv(fv=None, pmt=0, n=1, rate=0, due=False, compound_per_year=1):
    """
    Compute present value.
    Handles lump sum (pmt=0) and annuities.
    If compound_per_year == 0, simple discounting is used (only lump sum).
    """
    if compound_per_year == 0:
        # Simple discount (lump sum only)
        if pmt != 0:
            raise ValueError("Simple interest does not support payments.")
        return fv / (1 + rate * n)

    r = rate / compound_per_year
    n_periods = n * compound_per_year

    if fv is None:
        fv = 0
    # Present value of lump sum
    pv_lump = fv / (1 + r) ** n_periods

    if pmt == 0:
        return pv_lump

    # Present value of annuity
    if due:
        # Annuity due: payments at beginning
        pv_annuity = pmt * (1 + r) * (1 - (1 + r) ** -n_periods) / r
    else:
        # Ordinary annuity
        pv_annuity = pmt * (1 - (1 + r) ** -n_periods) / r

    return pv_lump + pv_annuity


def tvm_pmt(pv=None, fv=None, n=1, rate=0, due=False, compound_per_year=1):
    """
    Solve for periodic payment.
    Either pv or fv must be given (or both). If both are given, the payment
    will achieve the transition from pv to fv.
    """
    if compound_per_year == 0:
        raise ValueError("Payment calculation not supported for simple interest.")
    r = rate / compound_per_year
    n_periods = n * compound_per_year

    if pv is None and fv is None:
        raise ValueError("Either pv or fv must be provided.")
    if pv is None:
        pv = 0
    if fv is None:
        fv = 0

    # The relationship: fv = pv*(1+r)^n + pmt * factor
    if due:
        factor = (1 + r) * ((1 + r) ** n_periods - 1) / r
    else:
        factor = ((1 + r) ** n_periods - 1) / r

    # If we are solving for pmt given both pv and fv, the pmt makes up the difference.
    needed_fv = fv - pv * (1 + r) ** n_periods
    return needed_fv / factor


def tvm_n(pv=None, fv=None, pmt=0, rate=0, due=False, compound_per_year=1, guess=10):
    """
    Solve for number of years (n). Uses numerical root finding.
    At least one of pv, fv must be given. If pmt is given, annuity is assumed.
    """
    if compound_per_year == 0:
        # Simple interest: n = (fv/pv - 1) / rate
        if pmt != 0:
            raise ValueError("Simple interest cannot handle payments.")
        return (fv / pv - 1) / rate

    r = rate / compound_per_year

    if pmt == 0:
        # Lump sum: fv = pv * (1+r)^(n*compound_per_year)
        if pv is None or fv is None:
            raise ValueError("For lump sum, both pv and fv are required.")
        # n = log(fv/pv) / (compound_per_year * log(1+r))
        return np.log(fv / pv) / (compound_per_year * np.log(1 + r))

    # Annuity case: define function f(n) = target - value
    # We'll solve for n_periods first, then convert to years.
    def func(n_periods):
        # n_periods is total number of periods (float)
        if pv is not None and fv is not None:
            # Both given: pmt must bridge the difference
            target = fv - pv * (1 + r) ** n_periods
        elif pv is not None:
            target = -pv  # we want the annuity to bring pv to 0 at the end
        elif fv is not None:
            target = fv   # we want the annuity to accumulate to fv from 0
        else:
            raise ValueError("Need pv or fv for annuity.")

        if due:
            factor = (1 + r) * ((1 + r) ** n_periods - 1) / r
        else:
            factor = ((1 + r) ** n_periods - 1) / r
        return pmt * factor - target

    # Initial guess for n_periods
    n_periods_guess = guess * compound_per_year
    if HAS_SCIPY:
        # Use brentq if we can bracket, otherwise newton
        try:
            # Try to bracket: assume n between 0 and 1000 periods
            n_sol = brentq(func, 0, 1000)
        except (ValueError, RuntimeError):
            n_sol = newton(func, n_periods_guess)
    else:
        # Simple Newton-Raphson (crude)
        def func_prime(n_periods):
            # derivative of func w.r.t. n_periods
            if due:
                factor_prime = (1 + r) * ((1 + r) ** n_periods * np.log(1 + r)) / r
            else:
                factor_prime = ((1 + r) ** n_periods * np.log(1 + r)) / r
            return pmt * factor_prime
        n_sol = n_periods_guess
        for _ in range(100):
            f_val = func(n_sol)
            if abs(f_val) < 1e-8:
                break
            fp_val = func_prime(n_sol)
            if fp_val == 0:
                raise RuntimeError("Derivative zero; cannot solve.")
            n_sol -= f_val / fp_val
    return n_sol / compound_per_year


def tvm_rate(pv=None, fv=None, pmt=0, n=1, due=False, compound_per_year=1, guess=0.05):
    """
    Solve for annual interest rate. Uses numerical root finding.
    """
    if compound_per_year == 0:
        # Simple interest
        if pmt != 0:
            raise ValueError("Simple interest cannot handle payments.")
        return (fv / pv - 1) / n

    # For compound interest, we solve for per‑period rate, then annualize.
    def func(r_per):
        # r_per is rate per period
        n_periods = n * compound_per_year
        if pmt == 0:
            # Lump sum
            if pv is not None and fv is not None:
                return pv * (1 + r_per) ** n_periods - fv
            else:
                raise ValueError("For lump sum, need pv and fv.")
        else:
            # Annuity
            if pv is not None and fv is not None:
                # Both pv and fv given: pmt makes up difference
                target = fv - pv * (1 + r_per) ** n_periods
            elif pv is not None:
                target = -pv
            elif fv is not None:
                target = fv
            else:
                raise ValueError("Need pv or fv for annuity.")
            if due:
                factor = (1 + r_per) * ((1 + r_per) ** n_periods - 1) / r_per
            else:
                factor = ((1 + r_per) ** n_periods - 1) / r_per
            return pmt * factor - target

    r_per_guess = guess / compound_per_year
    if HAS_SCIPY:
        try:
            # Try to bracket between 0 and 1 (100% per period)
            r_per_sol = brentq(func, 0, 1)
        except (ValueError, RuntimeError):
            r_per_sol = newton(func, r_per_guess)
    else:
        # Simple Newton
        def func_prime(r_per):
            n_periods = n * compound_per_year
            if pmt == 0:
                return pv * n_periods * (1 + r_per) ** (n_periods - 1)
            else:
                # numerical derivative
                h = 1e-8
                return (func(r_per + h) - func(r_per - h)) / (2 * h)
        r_per_sol = r_per_guess
        for _ in range(100):
            f_val = func(r_per_sol)
            if abs(f_val) < 1e-8:
                break
            fp_val = func_prime(r_per_sol)
            if fp_val == 0:
                raise RuntimeError("Derivative zero; cannot solve.")
            r_per_sol -= f_val / fp_val
    return r_per_sol * compound_per_year


# ----------------------------------------------------------------------
# Schedule generation
# ----------------------------------------------------------------------
def annuity_schedule(pv=0, fv=0, pmt=0, n=1, rate=0, due=False, compound_per_year=1):
    """
    Generate a period‑by‑period schedule for an annuity (loan or investment).
    At least one of pv, fv, or pmt must be consistent. For a loan, pv is given and pmt is
    calculated if not provided. For an investment, fv may be given.
    Returns DataFrame with columns: Period, Start Balance, Payment, Interest, Principal, End Balance.
    """
    if compound_per_year == 0:
        raise ValueError("Schedule not supported for simple interest.")
    r = rate / compound_per_year
    n_periods = int(n * compound_per_year)

    # Determine missing variable if needed
    if pmt == 0:
        if pv != 0 and fv != 0:
            # Solve for pmt that takes pv to fv
            pmt = tvm_pmt(pv=pv, fv=fv, n=n, rate=rate, due=due, compound_per_year=compound_per_year)
        elif pv != 0:
            # Assume loan: we want to end at 0
            pmt = tvm_pmt(pv=pv, fv=0, n=n, rate=rate, due=due, compound_per_year=compound_per_year)
        elif fv != 0:
            # Assume investment: start at 0
            pmt = tvm_pmt(pv=0, fv=fv, n=n, rate=rate, due=due, compound_per_year=compound_per_year)
        else:
            raise ValueError("Need at least one of pv, fv, or pmt.")

    balance = pv
    data = []

    if due:
        # For annuity due, we need to adjust the schedule: first payment at time 0.
        # We'll implement a standard approach: treat the first period separately.
        # At time 0: payment reduces balance, then interest accrues on remaining for the first period.
        # Then subsequent periods are like ordinary annuity but with shifted balance.
        # We'll build the schedule with period numbers starting at 1.
        # Period 1: start = pv, payment at beginning, then interest on (start - pmt), end = (start - pmt)*(1+r)
        start = pv
        payment = pmt
        after_pmt = start - payment
        interest = after_pmt * r
        end = after_pmt + interest
        data.append([1, start, payment, interest, payment, end])
        balance = end
        for period in range(2, n_periods + 1):
            start = balance
            # For periods 2..n, payment at beginning? Actually for due, all payments are at beginning.
            # So at the start of period 2, we make another payment.
            after_pmt = start - payment
            interest = after_pmt * r
            end = after_pmt + interest
            data.append([period, start, payment, interest, payment, end])
            balance = end
    else:
        # Ordinary annuity: payment at end of each period
        for period in range(1, n_periods + 1):
            start = balance
            interest = start * r
            payment = pmt
            principal_paid = payment - interest
            balance += interest
            balance -= payment
            end = balance
            data.append([period, start, payment, interest, principal_paid, end])

    df = pd.DataFrame(data, columns=["Period", "Start Balance", "Payment", "Interest", "Principal Paid", "End Balance"])
    return df


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def format_currency(value):
    return f"${value:,.2f}"

def format_rate(value):
    return f"{value*100:.2f}%"

# ----------------------------------------------------------------------
# Interactive user interface
# ----------------------------------------------------------------------
def get_float(prompt, allow_zero=False, positive_only=False):
    while True:
        try:
            val = float(input(prompt))
            if positive_only and val <= 0:
                print("Please enter a positive number.")
                continue
            if not allow_zero and val == 0:
                print("Please enter a non-zero number (or enter 0 if allowed).")
                continue
            return val
        except ValueError:
            print("Invalid input. Enter a number.")

def get_int(prompt, min_val=0, allow_zero=False):
    while True:
        try:
            val = int(input(prompt))
            if val < min_val and not (allow_zero and val == 0):
                print(f"Please enter an integer >= {min_val} (or 0 if allowed).")
                continue
            return val
        except ValueError:
            print("Invalid input. Enter an integer.")

def get_yes_no(prompt):
    while True:
        ans = input(prompt + " (y/n): ").strip().lower()
        if ans in ('y','yes'):
            return True
        if ans in ('n','no'):
            return False
        print("Please answer y or n.")

def main():
    print("\n" + "="*60)
    print("   Comprehensive Time Value of Money (TVM) Solver")
    print("="*60)
    print("\nThis tool can solve for any TVM variable:")
    print("  - Future Value (FV)")
    print("  - Present Value (PV)")
    print("  - Payment (PMT)")
    print("  - Number of Periods (N)")
    print("  - Interest Rate (R)")
    print("It supports lump sums, ordinary annuities, annuities due, and simple interest.")
    print("For simple interest, set compounding frequency to 0 (payments not supported).\n")

    while True:
        print("\n--- Main Menu ---")
        print("1. Solve for Future Value (FV)")
        print("2. Solve for Present Value (PV)")
        print("3. Solve for Payment (PMT)")
        print("4. Solve for Number of Periods (N)")
        print("5. Solve for Interest Rate (R)")
        print("6. Generate Amortization/Investment Schedule")
        print("7. Exit")
        choice = input("Enter your choice (1-7): ").strip()

        if choice == '1':
            print("\n--- Solve for Future Value ---")
            pv = get_float("Present value (0 if none): ", allow_zero=True)
            pmt = get_float("Periodic payment (0 if none): ", allow_zero=True)
            n = get_float("Number of years: ", positive_only=True)
            rate = get_float("Annual interest rate (e.g., 0.05): ", positive_only=True)
            compound = get_int("Compounding periods per year (0 = simple interest): ", allow_zero=True)
            due = False
            if pmt != 0 and compound != 0:
                due = get_yes_no("Is this an annuity due (payments at beginning)?")
            try:
                fv = tvm_fv(pv=pv, pmt=pmt, n=n, rate=rate, due=due, compound_per_year=compound)
                print(f"\nFuture Value = {format_currency(fv)}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '2':
            print("\n--- Solve for Present Value ---")
            fv = get_float("Future value (0 if none): ", allow_zero=True)
            pmt = get_float("Periodic payment (0 if none): ", allow_zero=True)
            n = get_float("Number of years: ", positive_only=True)
            rate = get_float("Annual interest rate: ", positive_only=True)
            compound = get_int("Compounding periods per year (0 = simple interest): ", allow_zero=True)
            due = False
            if pmt != 0 and compound != 0:
                due = get_yes_no("Is this an annuity due?")
            try:
                pv = tvm_pv(fv=fv, pmt=pmt, n=n, rate=rate, due=due, compound_per_year=compound)
                print(f"\nPresent Value = {format_currency(pv)}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '3':
            print("\n--- Solve for Payment (PMT) ---")
            pv = get_float("Present value (0 if none): ", allow_zero=True)
            fv = get_float("Future value (0 if none): ", allow_zero=True)
            n = get_float("Number of years: ", positive_only=True)
            rate = get_float("Annual interest rate: ", positive_only=True)
            compound = get_int("Compounding periods per year (must be >=1 for payments): ", min_val=1)
            if pv == 0 and fv == 0:
                print("At least one of PV or FV must be non-zero.")
                continue
            due = get_yes_no("Is this an annuity due?")
            try:
                pmt = tvm_pmt(pv=pv, fv=fv, n=n, rate=rate, due=due, compound_per_year=compound)
                print(f"\nPeriodic Payment = {format_currency(pmt)}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '4':
            print("\n--- Solve for Number of Years (N) ---")
            pv = get_float("Present value (0 if none): ", allow_zero=True)
            fv = get_float("Future value (0 if none): ", allow_zero=True)
            pmt = get_float("Periodic payment (0 if none): ", allow_zero=True)
            rate = get_float("Annual interest rate: ", positive_only=True)
            compound = get_int("Compounding periods per year (0 = simple interest): ", allow_zero=True)
            due = False
            if pmt != 0 and compound != 0:
                due = get_yes_no("Is this an annuity due?")
            try:
                n = tvm_n(pv=pv, fv=fv, pmt=pmt, rate=rate, due=due, compound_per_year=compound)
                print(f"\nNumber of years = {n:.4f}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '5':
            print("\n--- Solve for Interest Rate (R) ---")
            pv = get_float("Present value (0 if none): ", allow_zero=True)
            fv = get_float("Future value (0 if none): ", allow_zero=True)
            pmt = get_float("Periodic payment (0 if none): ", allow_zero=True)
            n = get_float("Number of years: ", positive_only=True)
            compound = get_int("Compounding periods per year (0 = simple interest): ", allow_zero=True)
            due = False
            if pmt != 0 and compound != 0:
                due = get_yes_no("Is this an annuity due?")
            try:
                rate = tvm_rate(pv=pv, fv=fv, pmt=pmt, n=n, due=due, compound_per_year=compound)
                print(f"\nAnnual interest rate = {format_rate(rate)}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '6':
            print("\n--- Generate Schedule ---")
            print("This will create a period‑by‑period table for an annuity (loan or investment).")
            pv = get_float("Present value (0 if none): ", allow_zero=True)
            fv = get_float("Future value (0 if none): ", allow_zero=True)
            pmt = get_float("Periodic payment (enter 0 to auto‑calculate): ", allow_zero=True)
            n = get_float("Number of years: ", positive_only=True)
            rate = get_float("Annual interest rate: ", positive_only=True)
            compound = get_int("Compounding periods per year (must be >=1): ", min_val=1)
            due = get_yes_no("Is this an annuity due? (Warning: due schedule may be approximate)")
            try:
                schedule = annuity_schedule(pv=pv, fv=fv, pmt=pmt, n=n, rate=rate,
                                            due=due, compound_per_year=compound)
                print("\nSchedule (first 10 rows):")
                print(schedule.head(10))
                if get_yes_no("\nSave full schedule to CSV?"):
                    fname = input("Filename (e.g., schedule.csv): ").strip()
                    if not fname.endswith('.csv'):
                        fname += '.csv'
                    schedule.to_csv(fname, index=False)
                    print(f"Saved to {fname}")
                if get_yes_no("Show balance over time plot?"):
                    plt.figure(figsize=(10,6))
                    plt.plot(schedule["Period"], schedule["End Balance"], marker='o')
                    plt.xlabel("Period")
                    plt.ylabel("End Balance")
                    plt.title("Balance Over Time")
                    plt.grid(True)
                    plt.show()
                if get_yes_no("Show interest vs principal plot?"):
                    plt.figure(figsize=(12,6))
                    plt.bar(schedule["Period"], schedule["Principal Paid"], label="Principal", alpha=0.7)
                    plt.bar(schedule["Period"], schedule["Interest"], bottom=schedule["Principal Paid"],
                            label="Interest", alpha=0.7)
                    plt.xlabel("Period")
                    plt.ylabel("Amount")
                    plt.title("Interest vs Principal per Period")
                    plt.legend()
                    plt.grid(axis='y')
                    plt.show()
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '7':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    main()