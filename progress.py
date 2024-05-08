from alive_progress import alive_bar
import time

# retrieve your set of items

items = range(100)

# declare your expected total

with alive_bar(len(items)) as bar:

    # iterate as usual

    for item in items:

        # process each item

        bar()

        time.sleep(0.1)