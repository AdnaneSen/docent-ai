from docent.llm import get_llm


def main() -> None:
    llm = get_llm()

    print("CHAT :", llm.generate("Reply with exactly: Gemini is working."))

    vectors = llm.embed(["hello world"])
    dim = len(vectors[0])
    print(f"EMBED: dim={dim}  first3={[round(v, 4) for v in vectors[0][:3]]}")


if __name__ == "__main__":
    main()