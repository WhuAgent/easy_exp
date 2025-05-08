class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        self.token_num = 0
        self.token_cost = 0

    def to_openai_message(self):
        return {"role": self.role, "content": self.content}

    def __str__(self):
        return self.content

    def __repr__(self):
        repr_map = {
            "role": self.role,
            "content": self.content,
            "token": self.token_num,
            "cost": self.token_cost
        }
        return f"{repr_map}"

    def to_dict(self):
        repr_map = {
            "role": self.role,
            "content": self.content,
            "token": self.token_num,
            "cost": self.token_cost
        }
        return repr_map


class SystemMessage(Message):
    def __init__(self, content):
        super().__init__("system", content)


class UserMessage(Message):
    def __init__(self, content):
        super().__init__("user", content)


class AssistantMessage(Message):
    def __init__(self, content):
        super().__init__("assistant", content)


class OpenAIMessage(Message):
    def __init__(self, content, model, prompt_tokens, completion_tokens):
        super().__init__("assistant", content)

        self.model = model
        self.prompt_token_num = prompt_tokens
        self.prompt_token_cost = model_cost[self.model]["prompt_token"] * self.prompt_token_num / 1000
        self.completion_token_num = completion_tokens
        self.completion_token_cost = model_cost[self.model]["completion_token"] * self.completion_token_num / 1000
        self.token_num = self.prompt_token_num + self.completion_token_num
        self.token_cost = self.prompt_token_cost + self.completion_token_cost