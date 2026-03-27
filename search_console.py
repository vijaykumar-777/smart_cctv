from query_engine import QueryEngine
from video_player import play_event
from video_player import play_event

engine = QueryEngine()

print("AI CCTV Search Console")
print("Type 'exit' to quit")

while True:

    query = input("\nAsk something: ")

    if query.lower() == "exit":
        break

    results = engine.run_query(query)

    if not results:
        print("No results.")
        continue

    print("\nResults:")

    for i, r in enumerate(results):
        print(i, r)

    choice = input("\nSelect result number to play video (or press Enter): ")

    if choice.strip() == "":
        continue

    idx = int(choice)

    result = results[idx]

    if result["type"] == "intrusion":
        play_event(
    result["video_file"],
    result["video_time"],
    result["track_id"]
)

    elif result["type"] == "stay":
        play_from_timestamp(result["entry_time"])