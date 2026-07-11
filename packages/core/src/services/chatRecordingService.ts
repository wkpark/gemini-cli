/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  hasProperty,
  isStringProperty,
  isObjectProperty,
} from '../utils/checks.js';
import { type ThoughtSummary } from '../utils/thoughtUtils.js';
import { getProjectHash } from '../utils/paths.js';
import path from 'node:path';
import * as fs from 'node:fs';
import { sanitizeFilenamePart } from '../utils/fileUtils.js';
import { isNodeError } from '../utils/errors.js';
import {
  deleteSessionArtifactsAsync,
  deleteSubagentSessionDirAndArtifactsAsync,
} from '../utils/sessionOperations.js';
import { randomUUID } from 'node:crypto';
import type {
  Content,
  Part,
  PartListUnion,
  GenerateContentResponseUsageMetadata,
} from '@google/genai';
import { debugLogger } from '../utils/debugLogger.js';
import type { AgentLoopContext } from '../config/agent-loop-context.js';
import {
  SESSION_FILE_PREFIX,
  type TokensSummary,
  type ToolCallRecord,
  type ConversationRecordExtra,
  type MessageRecord,
  type ConversationRecord,
  type ResumedSessionData,
  type LoadConversationOptions,
  type RewindRecord,
  type SnapshotRecord,
  type MetadataUpdateRecord,
  type PartialMetadataRecord,
} from './chatRecordingTypes.js';
export * from './chatRecordingTypes.js';

/**
 * Warning message shown when recording is disabled due to disk full.
 */
const ENOSPC_WARNING_MESSAGE =
  'Chat recording disabled: No space left on device. ' +
  'The conversation will continue but will not be saved to disk. ' +
  'Free up disk space and restart to enable recording.';

function isRewindRecord(record: unknown): record is RewindRecord {
  return isStringProperty(record, '$rewindTo');
}

function isSnapshotRecord(record: unknown): record is SnapshotRecord {
  return (
    hasProperty(record, '$snapshot') &&
    record.$snapshot === true &&
    hasProperty(record, 'messages') &&
    Array.isArray(record.messages)
  );
}

function isMessageRecord(record: unknown): record is MessageRecord {
  return isStringProperty(record, 'id') && hasProperty(record, 'type');
}

function isMetadataUpdateRecord(
  record: unknown,
): record is MetadataUpdateRecord {
  return isObjectProperty(record, '$set');
}

function isPartialMetadataRecord(
  record: unknown,
): record is PartialMetadataRecord {
  return isStringProperty(record, 'sessionId') && isStringProperty(record, 'projectHash');
}

function isTextPart(part: unknown): part is { text: string } {
  return isStringProperty(part, 'text');
}

function isSessionIdRecord(record: unknown): record is { sessionId: string } {
  return isStringProperty(record, 'sessionId');
}

export async function loadConversationRecord(
  filePath: string,
  options?: LoadConversationOptions,
): Promise<
  | (ConversationRecord & {
      messageCount?: number;
      userMessageCount?: number;
      firstUserMessage?: string;
      hasUserOrAssistantMessage?: boolean;
      memoryScratchpadIsStale?: boolean;
    })
  | null
> {
  if (!fs.existsSync(filePath)) {
    return null;
  }

  try {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const lines = fileContent.split('\n');

    let metadata: Partial<ConversationRecord> = {};
    const messagesMap = new Map<string, MessageRecord>();
    const uniqueMessageIds = new Set<string>();
    let isTrackingMemoryScratchpadFreshness = false;
    let memoryScratchpadIsStale = false;
    let firstUserMessageStr: string | undefined;

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        // 1. FAST PATH for session browser metadata (very aggressive)
        if (options?.metadataOnly && trimmed.startsWith('{"id":')) {
          const idMatch = /^\{"id":"([^"]+)"/.exec(trimmed);
          if (idMatch) {
            const msgId = idMatch[1];
            if (!uniqueMessageIds.has(msgId)) {
              uniqueMessageIds.add(msgId);
              if (!firstUserMessageStr) {
                const typeMatch = /"type"\s*:\s*"(user|user_shell)"/.exec(
                  trimmed,
                );
                if (typeMatch) {
                  try {
                    const record = JSON.parse(trimmed) as unknown;
                    if (isMessageRecord(record) && record.content) {
                      if (Array.isArray(record.content)) {
                        firstUserMessageStr = record.content
                          .map((p) => (isTextPart(p) ? p.text : ''))
                          .join('');
                      } else if (typeof record.content === 'string') {
                        firstUserMessageStr = record.content;
                      }
                    }
                  } catch {
                    /* ignore */
                  }
                }
              }
            }
          }
          continue;
        }

        // 2. Metadata / Control Record Detection
        if (
          trimmed.includes('"sessionId"') &&
          trimmed.includes('"projectHash"')
        ) {
          try {
            const record = JSON.parse(trimmed) as unknown;
            if (isPartialMetadataRecord(record)) {
              metadata = { ...metadata, ...record };
              continue;
            }
          } catch {
            /* ignore */
          }
        }

        // 3. Regular Record Parsing
        const record = JSON.parse(trimmed) as unknown;
        if (isSnapshotRecord(record)) {
          const snapshot = record;
          // SAFETY: Only clear if snapshot actually has messages.
          if (
            !options?.metadataOnly &&
            snapshot.messages &&
            snapshot.messages.length > 0
          ) {
            messagesMap.clear();
          }
          for (const msg of snapshot.messages) {
            uniqueMessageIds.add(msg.id);
            if (!options?.metadataOnly) messagesMap.set(msg.id, msg);
            const isUser = msg.type === 'user' || msg.type === 'user_shell';
            if (!firstUserMessageStr && isUser && msg.content) {
              if (Array.isArray(msg.content)) {
                firstUserMessageStr = msg.content
                  .map((p) => (isTextPart(p) ? p.text : ''))
                  .join('');
              } else if (typeof msg.content === 'string') {
                firstUserMessageStr = msg.content;
              }
            }
          }
          memoryScratchpadIsStale = false;
        } else if (isRewindRecord(record)) {
          if (isTrackingMemoryScratchpadFreshness)
            memoryScratchpadIsStale = true;
          const rewindId = record.$rewindTo;
          if (!options?.metadataOnly) {
            let found = false;
            const idsToDelete: string[] = [];
            for (const [id] of messagesMap) {
              if (id === rewindId) found = true;
              if (found) idsToDelete.push(id);
            }
            if (found) {
              for (const id of idsToDelete) messagesMap.delete(id);
            } else {
              messagesMap.clear();
            }
          }
        } else if (isMessageRecord(record)) {
          memoryScratchpadIsStale = true;
          uniqueMessageIds.add(record.id);
          const isUser = record.type === 'user' || record.type === 'user_shell';
          if (!firstUserMessageStr && isUser && record.content) {
            if (Array.isArray(record.content)) {
              firstUserMessageStr = record.content
                .map((p) => (isTextPart(p) ? p.text : ''))
                .join('');
            } else if (typeof record.content === 'string') {
              firstUserMessageStr = record.content;
            }
          }
          if (!options?.metadataOnly) {
            messagesMap.set(record.id, record);
            if (
              options?.maxMessages &&
              messagesMap.size > options.maxMessages
            ) {
              const firstKey = messagesMap.keys().next().value;
              if (typeof firstKey === 'string') messagesMap.delete(firstKey);
            }
          }
        } else if (isMetadataUpdateRecord(record)) {
          if (hasProperty(record.$set, 'memoryScratchpad')) {
            isTrackingMemoryScratchpadFreshness = Boolean(
              record.$set.memoryScratchpad,
            );
            memoryScratchpadIsStale = false;
          }
          metadata = { ...metadata, ...record.$set };
        } else if (isPartialMetadataRecord(record)) {
          metadata = { ...metadata, ...record };
        }
      } catch {
        /* ignore parse errors */
      }
    }

    if (!metadata.sessionId || !metadata.projectHash) {
      return await parseLegacyRecordFallback(filePath, options);
    }

    const loadedMessages = Array.from(messagesMap.values());
    const userMessageCount = loadedMessages.filter(
      (m) => m.type === 'user' || m.type === 'user_shell',
    ).length;
    const hasUserOrAssistant = options?.metadataOnly
      ? true
      : loadedMessages.some(
          (m) =>
            m.type === 'user' || m.type === 'user_shell' || m.type === 'gemini',
        );

    return {
      sessionId: metadata.sessionId,
      projectHash: metadata.projectHash,
      startTime: metadata.startTime || new Date().toISOString(),
      lastUpdated: metadata.lastUpdated || new Date().toISOString(),
      summary: metadata.summary,
      memoryScratchpad: metadata.memoryScratchpad,
      directories: metadata.directories,
      kind: metadata.kind,
      messages: loadedMessages,
      messageCount: uniqueMessageIds.size,
      userMessageCount,
      hasUserOrAssistantMessage: hasUserOrAssistant,
      firstUserMessage: firstUserMessageStr,
      memoryScratchpadIsStale: isTrackingMemoryScratchpadFreshness
        ? memoryScratchpadIsStale
        : undefined,
    };
  } catch (error) {
    debugLogger.error('Error loading conversation record:', error);
    return null;
  }
}

export class ChatRecordingService {
  private conversationFile: string | null = null;
  private cachedConversation: ConversationRecord | null = null;
  private sessionId: string;
  private projectHash: string;
  private kind?: 'main' | 'subagent';
  private queuedThoughts: Array<ThoughtSummary & { timestamp: string }> = [];
  private queuedTokens: TokensSummary | null = null;
  private context: AgentLoopContext;

  constructor(context: AgentLoopContext) {
    this.context = context;
    this.sessionId = context.promptId;
    this.projectHash = getProjectHash(context.config.getProjectRoot());
  }

  async initialize(
    resumedSessionData?: ResumedSessionData,
    kind?: 'main' | 'subagent',
  ): Promise<void> {
    try {
      this.kind = kind;
      if (resumedSessionData) {
        this.conversationFile = resumedSessionData.filePath;
        this.sessionId = resumedSessionData.conversation.sessionId;
        this.kind = resumedSessionData.conversation.kind;

        const loadedRecord = await loadConversationRecord(
          this.conversationFile,
        );
        if (loadedRecord) {
          this.cachedConversation = loadedRecord;
          this.projectHash = this.cachedConversation.projectHash;

          if (this.conversationFile.endsWith('.json')) {
            // SAFETY: Never migrate an empty session to prevent accidental data loss.
            if (this.cachedConversation.messages.length === 0) {
              debugLogger.warn(
                'Skipping migration of empty .json session to prevent data loss.',
              );
            } else {
              this.conversationFile = this.conversationFile + 'l'; // e.g. session-foo.jsonl

              // Migrate the entire legacy record to the new file
              const initialMetadata = {
                sessionId: this.sessionId,
                projectHash: this.projectHash,
                startTime: this.cachedConversation.startTime,
                lastUpdated: this.cachedConversation.lastUpdated,
                kind: this.cachedConversation.kind,
                directories: this.cachedConversation.directories,
                summary: this.cachedConversation.summary,
              };
              this.appendRecord(initialMetadata);
              for (const msg of this.cachedConversation.messages) {
                this.appendRecord(msg);
              }
              if (this.cachedConversation.memoryScratchpad) {
                this.appendRecord({
                  $set: {
                    memoryScratchpad: this.cachedConversation.memoryScratchpad,
                  },
                });
              }
            }
          }

          // Update the session ID in the existing file
          this.updateMetadata({ sessionId: this.sessionId });
        } else {
          throw new Error('Failed to load resumed session data from file');
        }
      } else {
        // Create new session
        this.sessionId = this.context.promptId;
        let chatsDir = path.join(
          this.context.config.storage.getProjectTempDir(),
          'chats',
        );

        // subagents are nested under the complete parent session id
        if (this.kind === 'subagent' && this.context.parentSessionId) {
          const safeParentId = sanitizeFilenamePart(
            this.context.parentSessionId,
          );
          if (!safeParentId) {
            throw new Error(
              `Invalid parentSessionId after sanitization: ${this.context.parentSessionId}`,
            );
          }
          chatsDir = path.join(chatsDir, safeParentId);
        }

        fs.mkdirSync(chatsDir, { recursive: true });

        const timestamp = new Date()
          .toISOString()
          .slice(0, 16)
          .replace(/:/g, '-');
        const safeSessionId = sanitizeFilenamePart(this.sessionId);
        if (!safeSessionId) {
          throw new Error(
            `Invalid sessionId after sanitization: ${this.sessionId}`,
          );
        }

        let filename: string;
        if (this.kind === 'subagent') {
          filename = `${safeSessionId}.jsonl`;
        } else {
          filename = `${SESSION_FILE_PREFIX}${timestamp}-${safeSessionId.slice(
            0,
            8,
          )}.jsonl`;
        }
        this.conversationFile = path.join(chatsDir, filename);

        const directories =
          this.kind === 'subagent'
            ? [
                ...(this.context.config
                  .getWorkspaceContext()
                  ?.getDirectories() ?? []),
              ]
            : undefined;

        const initialMetadata = {
          sessionId: this.sessionId,
          projectHash: this.projectHash,
          startTime: new Date().toISOString(),
          lastUpdated: new Date().toISOString(),
          kind: this.kind,
          directories,
        };

        this.appendRecord(initialMetadata);
        this.cachedConversation = {
          ...initialMetadata,
          messages: [],
        };
      }

      this.queuedThoughts = [];
      this.queuedTokens = null;
    } catch (error) {
      if (isNodeError(error) && error.code === 'ENOSPC') {
        this.conversationFile = null;
        debugLogger.warn(ENOSPC_WARNING_MESSAGE);
        return;
      }
      debugLogger.error('Error initializing chat recording service:', error);
      throw error;
    }
  }

  /**
   * Records a snapshot of the conversation state.
   * This marks a clean starting point (e.g., after compression) and
   * contains the full set of messages that should be active from this point.
   */
  recordSnapshot(messages: MessageRecord[]): void {
    if (!this.conversationFile || !this.cachedConversation) {
      return;
    }
    this.cachedConversation.messages = [...messages];
    this.appendRecord({ $snapshot: true, messages });
  }

  private appendRecord(record: unknown): void {
    if (!this.conversationFile) return;
    try {
      const line = JSON.stringify(record) + '\n';
      fs.mkdirSync(path.dirname(this.conversationFile), { recursive: true });
      fs.appendFileSync(this.conversationFile, line);
    } catch (error) {
      if (isNodeError(error) && error.code === 'ENOSPC') {
        this.conversationFile = null;
        debugLogger.warn(ENOSPC_WARNING_MESSAGE);
      } else {
        throw error;
      }
    }
  }

  private updateMetadata(updates: Partial<ConversationRecord>): void {
    if (!this.cachedConversation) return;

    // PERF: Skip recording if the updates don't actually change anything.
    const changedFields = Object.entries(updates).filter(([key, value]) => {
      const isKeyOfConversation = (
        k: string,
      ): k is keyof typeof this.cachedConversation =>
        k in this.cachedConversation!;

      if (isKeyOfConversation(key)) {
        const current = this.cachedConversation![key];
        return JSON.stringify(current) !== JSON.stringify(value);
      }
      return true;
    });

    if (changedFields.length === 0) {
      return;
    }

    Object.assign(this.cachedConversation, updates);
    this.appendRecord({ $set: updates });
  }

  private pushMessage(msg: MessageRecord): void {
    if (!this.cachedConversation) return;

    // We append the full message to the log
    this.appendRecord(msg);

    // Now update memory
    const index = this.cachedConversation.messages.findIndex(
      (m) => m.id === msg.id,
    );
    if (index !== -1) {
      this.cachedConversation.messages[index] = msg;
    } else {
      this.cachedConversation.messages.push(msg);
    }
  }

  private getLastMessage(
    conversation: ConversationRecord,
  ): MessageRecord | undefined {
    return conversation.messages.at(-1);
  }

  private newMessage(
    type: ConversationRecordExtra['type'],
    content: PartListUnion,
    displayContent?: PartListUnion,
  ): MessageRecord {
    return {
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      type,
      content,
      displayContent,
    };
  }

  recordMessage(message: {
    model: string | undefined;
    type: ConversationRecordExtra['type'];
    content: PartListUnion;
    displayContent?: PartListUnion;
  }): void {
    if (!this.conversationFile || !this.cachedConversation) return;

    try {
      const msg = this.newMessage(
        message.type,
        message.content,
        message.displayContent,
      );
      if (msg.type === 'gemini') {
        msg.thoughts = this.queuedThoughts;
        msg.tokens = this.queuedTokens;
        msg.model = message.model;
        this.queuedThoughts = [];
        this.queuedTokens = null;
      }
      this.pushMessage(msg);
      this.updateMetadata({ lastUpdated: new Date().toISOString() });
    } catch (error) {
      debugLogger.error('Error saving message to chat history.', error);
      throw error;
    }
  }

  recordThought(thought: ThoughtSummary): void {
    if (!this.conversationFile) return;
    this.queuedThoughts.push({
      ...thought,
      timestamp: new Date().toISOString(),
    });
  }

  recordMessageTokens(
    respUsageMetadata: GenerateContentResponseUsageMetadata,
  ): void {
    if (!this.conversationFile || !this.cachedConversation) return;

    try {
      const tokens = {
        input: respUsageMetadata.promptTokenCount ?? 0,
        output: respUsageMetadata.candidatesTokenCount ?? 0,
        cached: respUsageMetadata.cachedContentTokenCount ?? 0,
        thoughts: respUsageMetadata.thoughtsTokenCount ?? 0,
        tool: respUsageMetadata.toolUsePromptTokenCount ?? 0,
        total: respUsageMetadata.totalTokenCount ?? 0,
      };
      const lastMsg = this.getLastMessage(this.cachedConversation);
      if (lastMsg && lastMsg.type === 'gemini' && !lastMsg.tokens) {
        lastMsg.tokens = tokens;
        this.queuedTokens = null;
        this.pushMessage(lastMsg);
      } else {
        this.queuedTokens = tokens;
      }
    } catch (error) {
      debugLogger.error(
        'Error updating message tokens in chat history.',
        error,
      );
      throw error;
    }
  }

  recordMessageUsage(usage: GenerateContentResponseUsageMetadata): void {
    this.recordMessageTokens(usage);
  }

  recordToolCalls(model: string, toolCalls: ToolCallRecord[]): void {
    if (!this.conversationFile || !this.cachedConversation) return;

    const toolRegistry = this.context.toolRegistry;
    const enrichedToolCalls = toolCalls.map((toolCall) => {
      const toolInstance = toolRegistry.getTool(toolCall.name);
      return {
        ...toolCall,
        displayName: toolInstance?.displayName || toolCall.name,
        description:
          toolCall.description?.trim() || toolInstance?.description || '',
        renderOutputAsMarkdown: toolInstance?.isOutputMarkdown || false,
      };
    });

    try {
      const lastMsg = this.getLastMessage(this.cachedConversation);
      if (
        !lastMsg ||
        lastMsg.type !== 'gemini' ||
        this.queuedThoughts.length > 0
      ) {
        const newMsg: MessageRecord = {
          ...this.newMessage('gemini' as const, ''),
          type: 'gemini' as const,
          toolCalls: enrichedToolCalls,
          thoughts: this.queuedThoughts,
          model,
        };
        if (this.queuedThoughts.length > 0) {
          newMsg.thoughts = this.queuedThoughts;
          this.queuedThoughts = [];
        }
        if (this.queuedTokens) {
          newMsg.tokens = this.queuedTokens;
          this.queuedTokens = null;
        }
        this.pushMessage(newMsg);
      } else {
        if (!lastMsg.toolCalls) {
          lastMsg.toolCalls = [];
        }
        // Deep clone toolCalls to avoid modifying memory references directly
        const updatedToolCalls = [...lastMsg.toolCalls];

        for (const toolCall of enrichedToolCalls) {
          const index = updatedToolCalls.findIndex(
            (tc) => tc.id === toolCall.id,
          );
          if (index !== -1) {
            updatedToolCalls[index] = {
              ...updatedToolCalls[index],
              ...toolCall,
            };
          } else {
            updatedToolCalls.push(toolCall);
          }
        }

        lastMsg.toolCalls = updatedToolCalls;
        this.pushMessage(lastMsg);
      }
    } catch (error) {
      debugLogger.error(
        'Error adding tool call to message in chat history.',
        error,
      );
      throw error;
    }
  }

  saveSummary(summary: string): void {
    if (!this.conversationFile) return;
    try {
      this.updateMetadata({ summary });
    } catch (error) {
      debugLogger.error('Error saving summary to chat history.', error);
    }
  }

  recordDirectories(directories: readonly string[]): void {
    if (!this.conversationFile) return;
    try {
      this.updateMetadata({ directories: [...directories] });
    } catch (error) {
      debugLogger.error('Error saving directories to chat history.', error);
    }
  }

  getConversation(): ConversationRecord | null {
    if (!this.conversationFile) return null;
    return this.cachedConversation;
  }

  getConversationFilePath(): string | null {
    return this.conversationFile;
  }

  /**
   * Deletes a session file by sessionId, filename, or basename.
   * Derives an 8-character shortId to find and delete all associated files
   * (parent and subagents).
   *
   * @throws {Error} If shortId validation fails.
   */
  async deleteSession(sessionIdOrBasename: string): Promise<void> {
    try {
      const tempDir = this.context.config.storage.getProjectTempDir();
      const chatsDir = path.join(tempDir, 'chats');
      const shortId = this.deriveShortId(sessionIdOrBasename);

      // Using stat instead of existsSync for async sanity
      if (!(await fs.promises.stat(chatsDir).catch(() => null))) {
        return; // Nothing to delete
      }

      const matchingFiles = await this.getMatchingSessionFiles(
        chatsDir,
        shortId,
      );
      for (const file of matchingFiles) {
        await this.deleteSessionAndArtifacts(chatsDir, file, tempDir);
      }
    } catch (error) {
      debugLogger.error('Error deleting session file.', error);
      throw error;
    }
  }

  private deriveShortId(sessionIdOrBasename: string): string {
    let shortId = sessionIdOrBasename;
    if (sessionIdOrBasename.startsWith(SESSION_FILE_PREFIX)) {
      const withoutExt = sessionIdOrBasename.replace(/\.jsonl?$/, '');
      const parts = withoutExt.split('-');
      shortId = parts[parts.length - 1];
    } else if (sessionIdOrBasename.length >= 8) {
      shortId = sessionIdOrBasename.slice(0, 8);
    } else {
      throw new Error('Invalid sessionId or basename provided for deletion');
    }

    if (shortId.length !== 8) {
      throw new Error('Derived shortId must be exactly 8 characters');
    }

    return shortId;
  }

  private async getMatchingSessionFiles(
    chatsDir: string,
    shortId: string,
  ): Promise<string[]> {
    const files = await fs.promises.readdir(chatsDir);
    return files.filter(
      (f) =>
        f.startsWith(SESSION_FILE_PREFIX) &&
        (f.endsWith(`-${shortId}.json`) || f.endsWith(`-${shortId}.jsonl`)),
    );
  }

  /**
   * Deletes a single session file and its associated logs, tool-outputs, and directory.
   */
  private async deleteSessionAndArtifacts(
    chatsDir: string,
    file: string,
    tempDir: string,
  ): Promise<void> {
    const filePath = path.join(chatsDir, file);
    let fullSessionId: string | undefined;

    try {
      const CHUNK_SIZE = 4096;
      const buffer = Buffer.alloc(CHUNK_SIZE);
      let firstLine: string;
      let fd: fs.promises.FileHandle | undefined;
      try {
        fd = await fs.promises.open(filePath, 'r');
        const { bytesRead } = await fd.read(buffer, 0, CHUNK_SIZE, 0);
        if (bytesRead > 0) {
          const contentChunk = buffer.toString('utf8', 0, bytesRead);
          const newlineIndex = contentChunk.indexOf('\n');
          firstLine =
            newlineIndex !== -1
              ? contentChunk.substring(0, newlineIndex)
              : contentChunk;

          try {
            const content = JSON.parse(firstLine) as unknown;
            if (isSessionIdRecord(content)) {
              fullSessionId = content.sessionId;
            }
          } catch {
            // If first line parse fails, it might be a legacy pretty-printed JSON.
            // We'll fall back to full file read below.
          }
        }
      } finally {
        if (fd !== undefined) {
          await fd.close();
        }
      }

      // Fallback for legacy JSON files if we couldn't get sessionId from first line
      if (!fullSessionId) {
        try {
          const fileContent = await fs.promises.readFile(filePath, 'utf8');
          const parsed = JSON.parse(fileContent) as unknown;
          if (isSessionIdRecord(parsed)) {
            fullSessionId = parsed.sessionId;
          }
        } catch {
          // Ignore parse errors, we'll still try to unlink the file
        }
      }

      if (fullSessionId) {
        // Delegate to shared utility!
        await deleteSessionArtifactsAsync(fullSessionId, tempDir);
        await deleteSubagentSessionDirAndArtifactsAsync(
          fullSessionId,
          chatsDir,
          tempDir,
        );
      }
    } catch (error) {
      debugLogger.error(
        `Error deleting artifacts for session file ${file}:`,
        error,
      );
    } finally {
      // ALWAYS try to delete the session file itself
      try {
        // SAFETY: Never physically delete session files. Move to .bak instead for recovery.
        debugLogger.warn(
          `[SAFETY] Moving session file to .bak instead of deleting: ${filePath}`,
        );
        await fs.promises.rename(filePath, `${filePath}.bak`).catch(() =>
          // Fallback to delete only if rename fails and it's absolutely necessary
          fs.promises.unlink(filePath).catch(() => {}),
        );
      } catch (error) {
        if (isNodeError(error) && error.code !== 'ENOENT') {
          debugLogger.error(`Error unlinking session file ${file}:`, error);
        }
      }
    }
  }

  /**
   * Asynchronously deletes the current session's chat file and tool outputs.
   * This encapsulates the session ID logic and uses non-blocking I/O to avoid
   * blocking the event loop on exit.
   */
  async deleteCurrentSessionAsync(): Promise<void> {
    if (!this.conversationFile) {
      return;
    }

    try {
      const tempDir = this.context.config.storage.getProjectTempDir();

      // SAFETY: Never physically delete current session file. Move to .bak instead.
      debugLogger.warn(
        `[SAFETY] Moving current session file to .bak instead of deleting: ${this.conversationFile}`,
      );
      await fs.promises
        .rename(this.conversationFile, `${this.conversationFile}.bak`)
        .catch(() => {
          // File may not exist; ignore.
        });

      // Delegate tool-output and log cleanup to the shared utility.
      await deleteSessionArtifactsAsync(this.sessionId, tempDir);
    } catch (error) {
      debugLogger.error('Error deleting current session.', error);
      throw error;
    }
  }

  /**
   * Rewinds the conversation to the state just before the specified message ID.
   * All messages from (and including) the specified ID onwards are removed.
   */
  rewindTo(messageId: string): ConversationRecord | null {
    if (!this.conversationFile || !this.cachedConversation) return null;

    const messageIndex = this.cachedConversation.messages.findIndex(
      (m) => m.id === messageId,
    );

    if (messageIndex === -1) {
      debugLogger.error(
        'Message to rewind to not found in conversation history',
      );
      return this.cachedConversation;
    }

    this.cachedConversation.messages = this.cachedConversation.messages.slice(
      0,
      messageIndex,
    );
    this.appendRecord({ $rewindTo: messageId });
    return this.cachedConversation;
  }

  updateMessagesFromHistory(history: readonly Content[]): void {
    if (!this.conversationFile || !this.cachedConversation) return;

    try {
      const partsMap = new Map<string, Part[]>();
      for (const content of history) {
        if (content.role === 'user' && content.parts) {
          const callIds = content.parts
            .map((p) => p.functionResponse?.id)
            .filter((id): id is string => !!id);

          if (callIds.length === 0) continue;

          let currentCallId = callIds[0];
          for (const part of content.parts) {
            if (part.functionResponse?.id) {
              currentCallId = part.functionResponse.id;
            }

            if (!partsMap.has(currentCallId)) {
              partsMap.set(currentCallId, []);
            }
            partsMap.get(currentCallId)!.push(part);
          }
        }
      }

      for (const message of this.cachedConversation.messages) {
        let msgChanged = false;
        if (message.type === 'gemini' && message.toolCalls) {
          for (const toolCall of message.toolCalls) {
            const newParts = partsMap.get(toolCall.id);
            if (newParts !== undefined) {
              toolCall.result = newParts;
              msgChanged = true;
            }
          }
        }
        if (msgChanged) {
          // Push updated message to log
          this.pushMessage(message);
        }
      }
    } catch (error) {
      debugLogger.error(
        'Error updating conversation history from memory.',
        error,
      );
      throw error;
    }
  }
}

async function parseLegacyRecordFallback(
  filePath: string,
  options?: LoadConversationOptions,
): Promise<
  | (ConversationRecord & {
      messageCount?: number;
      userMessageCount?: number;
      firstUserMessage?: string;
      hasUserOrAssistantMessage?: boolean;
    })
  | null
> {
  try {
    const fileContent = await fs.promises.readFile(filePath, 'utf8');
    const parsed = JSON.parse(fileContent) as unknown;

    const isLegacyRecord = (val: unknown): val is ConversationRecord =>
      typeof val === 'object' && val !== null && 'sessionId' in val;

    if (isLegacyRecord(parsed)) {
      const legacyRecord = parsed;
      if (options?.metadataOnly) {
        let fallbackFirstUserMessageStr: string | undefined;
        const firstUserMessage = legacyRecord.messages?.find(
          (m) => m.type === 'user',
        );
        if (firstUserMessage) {
          const rawContent = firstUserMessage.content;
          if (Array.isArray(rawContent)) {
            fallbackFirstUserMessageStr = rawContent
              .map((p: unknown) => (isTextPart(p) ? p['text'] : ''))
              .join('');
          } else if (typeof rawContent === 'string') {
            fallbackFirstUserMessageStr = rawContent;
          }
        }
        return {
          ...legacyRecord,
          messages: [],
          messageCount: legacyRecord.messages?.length || 0,
          userMessageCount:
            legacyRecord.messages?.filter((m) => m.type === 'user').length || 0,
          firstUserMessage: fallbackFirstUserMessageStr,
          hasUserOrAssistantMessage:
            legacyRecord.messages?.some(
              (m) => m.type === 'user' || m.type === 'gemini',
            ) || false,
        };
      }
      return {
        ...legacyRecord,
        userMessageCount:
          legacyRecord.messages?.filter((m) => m.type === 'user').length || 0,
        hasUserOrAssistantMessage:
          legacyRecord.messages?.some(
            (m) => m.type === 'user' || m.type === 'gemini',
          ) || false,
      };
    }
  } catch {
    // ignore legacy fallback parse error
  }
  return null;
}
