import time

def countdown():
    for i in range(100, 0, -1):
        print(i)
        time.sleep(0.75)  # Sleep for half a second
    
    print("Countdown complete!")

if __name__ == "__main__":
    print("Starting countdown from 100 to 1...")
    countdown()
