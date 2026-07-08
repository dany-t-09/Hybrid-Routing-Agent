from classifier import classify_task
from solver import solve


def main() -> None:
    query = input("Ask the general agent: ").strip()
    if not query:
        print("Please enter a question or task.")
        return

    task_type = classify_task(query)
    answer = solve(query, task_type)
    print(answer)


if __name__ == "__main__":
    main()
