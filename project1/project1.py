class Chatbot:
    def __init__(self):
        self.responses = {
            "hello": "Hi there! How can I help you?",
            "how are you": "I am doing great, thank you!",
            "what is your name": "I am DecodeLabs Bot!",
            "what can you do": "I can answer basic questions.",
            "bye": "Goodbye! Have a nice day!",
            "who made you": "Jawad made me",
            "in which language you were coded": "I was coded in python"
        }

    def sanitize(self, user_input):
        return user_input.lower().strip()

    def get_response(self, user_input):
        clean = self.sanitize(user_input)
        return self.responses.get(clean, "I dont know!")

    def run(self):
        print("Bot: Hello! Type 'quit' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower().strip() == "quit" or user_input.lower().strip() == "exit":
                print("Bot: Goodbye!")
                break
            reply = self.get_response(user_input)
            print("Bot:", reply)

bot = Chatbot()
bot.run()




