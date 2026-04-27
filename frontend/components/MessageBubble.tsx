import { Message } from "@/lib/graphql/types";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-2 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      <Avatar className="h-7 w-7 shrink-0">
        <AvatarFallback className="text-xs">
          {isUser ? "You" : "🎵"}
        </AvatarFallback>
      </Avatar>
      <div
        className={`max-w-[80%] rounded-xl px-3 py-2 text-sm ${
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}
