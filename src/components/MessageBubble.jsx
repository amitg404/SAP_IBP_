// MessageBubble — renders a single chat message (user or agent)
// props: { role: 'user'|'agent'|'error', content: string, timestamp: string, chart?: object, persona?: string }
import React from 'react';
import ChartWidget from './ChartWidget';
import { User, Activity, Bot, AlertTriangle } from 'lucide-react';

export default function MessageBubble({ role, content, timestamp, chart, persona }) {
  const isUser = role === 'user';
  const isError = role === 'error';

  // Determine avatar and styling based on persona
  let AvatarIcon = Bot;
  let bubbleClass = 'billy';
  let avatarClass = 'billy-avatar';

  if (isUser) {
    AvatarIcon = User;
    bubbleClass = 'user';
    avatarClass = 'user-avatar';
  } else if (isError) {
    AvatarIcon = AlertTriangle;
    bubbleClass = 'error';
    avatarClass = 'error-avatar';
  } else {
    if (persona === 'blake') {
      AvatarIcon = Activity;
      bubbleClass = 'blake-bubble';
      avatarClass = 'blake-avatar';
    } else if (persona === 'chris') {
      AvatarIcon = Bot; // Or use another icon for Chris like TrendingUp
      bubbleClass = 'chris-bubble';
      avatarClass = 'chris-avatar';
    } else {
      AvatarIcon = Bot; // Router
      bubbleClass = 'router-bubble';
      avatarClass = 'router-avatar';
    }
  }

  return (
    <div className={`message-row ${isUser ? 'user' : 'billy'}`}>
      <div className={`message-avatar ${avatarClass}`}>
        <AvatarIcon size={18} strokeWidth={2.5} />
      </div>
      <div className="message-content">
        <div className={`bubble ${bubbleClass}`}>
          {content}
        </div>
        {/* Chart rendered below the text bubble, only when present */}
        {chart && <ChartWidget chart={chart} />}
        {timestamp && (
          <div className="message-time">
            {!isUser && persona && (
              <span className="persona-label">{persona.toUpperCase()} • </span>
            )}
            {timestamp}
          </div>
        )}
      </div>
    </div>
  );
}
