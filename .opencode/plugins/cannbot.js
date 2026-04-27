/**
 * CANNBot plugin for OpenCode.ai
 *
 * Registers CANNBot skills for auto-discovery and injects AGENTS.md
 * orchestration context into the first user message of each session.
 *
 * Supports multiple teams via plugin options:
 *   ["cannbot@...", {"team": "all"}]                        — install ALL agents & skills
 *   ["cannbot@..."]                                        — install ops-direct-invoke only (default)
 *   ["cannbot@...", {"team": "ops-direct-invoke"}]         — install only this team's deps
 *   ["cannbot@...", {"team": "pypto-op-orchestrator"}]     — install only this team's deps
 *
 * Each team declares its agents in .claude-plugin/plugin.json:
 *   { "agents": ["./agents/name.md"], "dependencies": ["skills-plugin-name"] }
 * Skills are resolved via marketplace.json using the dependencies field.
 * When team is "all", all agents and skills from all teams are installed.
 */

import path from 'path';
import fs from 'fs';
import os from 'os';
import { fileURLToPath } from 'url';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../..');
const TEAMS_DIR = path.join(REPO_ROOT, 'plugins-official');
const SHARED_AGENTS_DIR = path.join(REPO_ROOT, 'ops/agents');
const SHARED_SKILLS_DIR = path.join(REPO_ROOT, 'ops/skills');

const DEFAULT_TEAM = 'ops-direct-invoke';
const CONTEXT_TAG = 'CANNBOT_CONTEXT';
const CONTEXT_TAG_OPEN = `<${CONTEXT_TAG}>`;

// teamName comes from user-supplied plugin options. Restrict it to a safe
// subset so it cannot be used for path traversal or argv injection.
const TEAM_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;

const warn = (msg) => {
  try { console.error(`[CANNBot] ${msg}`); } catch { /* ignore */ }
};

/**
 * Resolve the on-disk location where per-team external repositories (asc-devkit,
 * pypto) are cloned when the plugin is installed via `opencode plugin
 * cannbot@git+...`. External data lives here — separate from the plugin source
 * tree (`~/.cache/opencode/packages/cannbot.../`) so clones never pollute the
 * OpenCode plugin cache. The `init.sh` flow (where the user has their own
 * repo checkout) writes to `$SCRIPT_DIR/asc-devkit` instead and is unaffected.
 *
 * Precedence:
 *   1. $CANNBOT_DATA_DIR (user override)
 *   2. $XDG_CACHE_HOME/cannbot
 *   3. $HOME/.cache/cannbot
 *   4. os.tmpdir()/cannbot  (last-resort fallback for exotic envs)
 *
 * Each team gets its own subdirectory: <root>/<team>/{asc-devkit,pypto}.
 */
const cannbotDataRoot = () => {
  const override = process.env.CANNBOT_DATA_DIR;
  if (override && path.isAbsolute(override)) return override;
  const xdg = process.env.XDG_CACHE_HOME;
  if (xdg && path.isAbsolute(xdg)) return path.join(xdg, 'cannbot');
  const home = process.env.HOME || os.homedir();
  if (home) return path.join(home, '.cache', 'cannbot');
  return path.join(os.tmpdir(), 'cannbot');
};

const teamCacheDir = (teamName) => path.join(cannbotDataRoot(), teamName);

/**
 * Check whether teamName is a safe, bounded identifier.
 */
const isValidTeamName = (name) =>
  typeof name === 'string' && TEAM_NAME_RE.test(name);

/**
 * Check whether `dir` looks like a usable git clone (has .git/HEAD readable).
 * Used to detect partial/interrupted clones.
 */
const isUsableClone = (dir) => {
  try {
    return fs.statSync(path.join(dir, '.git', 'HEAD')).isFile();
  } catch {
    return false;
  }
};

/**
 * Best-effort recursive removal, tolerating absent paths and odd file modes.
 */
const rmRfQuiet = (target) => {
  try {
    fs.rmSync(target, { recursive: true, force: true });
  } catch (e) {
    warn(`failed to remove ${target}: ${e.message}`);
  }
};

/**
 * Check whether the cannbot plugin is explicitly declared in any
 * `opencode.json` the current session would load — either the
 * project-local file at `<directory>/.opencode/opencode.json` or the
 * global file at `~/.config/opencode/opencode.json`.
 *
 * Why: OpenCode auto-discovers `<repo>/.opencode/plugins/*.js` from any
 * git-root it finds, regardless of whether the plugin was actually
 * installed. When a user opens the cannbot source tree itself (or any
 * repo that happens to vendor this file), OpenCode loads this plugin
 * and would otherwise register agents / inject context without the user
 * ever asking for it. Requiring a declaration in an `opencode.json`
 * means the user must opt in explicitly.
 *
 * A declaration is any entry in the top-level `plugin` array that
 * begins with `cannbot@` (the form `opencode plugin install` writes).
 * Entries may be a bare string, or a `[spec, options]` tuple when the
 * user passes plugin options such as `{"team": "all"}`.
 */
const CANNBOT_SPEC_RE = /^cannbot@/;
const isCannbotEntry = (entry) => {
  if (typeof entry === 'string') return CANNBOT_SPEC_RE.test(entry);
  if (Array.isArray(entry) && typeof entry[0] === 'string') return CANNBOT_SPEC_RE.test(entry[0]);
  return false;
};
const isPluginDeclared = (directory) => {
  const candidates = [];
  if (directory) {
    candidates.push(path.join(directory, '.opencode', 'opencode.json'));
  }
  const home = process.env.HOME || os.homedir();
  if (home) {
    candidates.push(path.join(home, '.config', 'opencode', 'opencode.json'));
  }
  for (const cfgPath of candidates) {
    let text;
    try { text = fs.readFileSync(cfgPath, 'utf8'); } catch { continue; }
    let cfg;
    try { cfg = JSON.parse(text); } catch { continue; }
    const plugins = Array.isArray(cfg && cfg.plugin) ? cfg.plugin : [];
    if (plugins.some(isCannbotEntry)) {
      return true;
    }
  }
  return false;
};

/**
 * Remove a dst symlink iff it is a symlink pointing under allowedPrefix.
 * Never touches real files/dirs or symlinks pointing elsewhere. The prefix
 * match is anchored at a path separator so `/x/skills` does not match
 * `/x/skillsBAD/...`.
 */
const safeUnlinkStaleSymlink = (dst, allowedPrefix) => {
  let stat;
  try { stat = fs.lstatSync(dst); } catch { return; }
  if (!stat.isSymbolicLink()) return;
  let target;
  try { target = fs.readlinkSync(dst); } catch { return; }
  if (target !== allowedPrefix && !target.startsWith(allowedPrefix + path.sep)) return;
  try { fs.unlinkSync(dst); }
  catch (e) { warn(`failed to unlink stale symlink ${dst}: ${e.message}`); }
};

/**
 * Load all agents/skills from the shared pool and all team directories.
 * Returns { agents: string[], skills: string[] }.
 *
 * Agents are collected by scanning every team directory under plugins-official/
 * because the shared ops/agents pool was emptied during the cd5b220 refactor.
 * Skills are taken from the ops/skills shared pool (no per-team skills dirs exist).
 */
const loadAllDeps = () => {
  let agents = [];
  let skills = [];

  // Scan all team directories for agents (post-cd5b220 architecture)
  try {
    const teams = fs.readdirSync(TEAMS_DIR).filter(d => {
      const agentsDir = path.join(TEAMS_DIR, d, 'agents');
      return fs.existsSync(agentsDir) && fs.statSync(agentsDir).isDirectory();
    });
    for (const team of teams) {
      const teamAgentsDir = path.join(TEAMS_DIR, team, 'agents');
      const teamAgents = fs.readdirSync(teamAgentsDir)
        .filter(f => f.endsWith('.md'))
        .map(f => f.replace(/\.md$/, ''));
      agents = agents.concat(teamAgents);
    }
    agents = [...new Set(agents)];
  } catch (e) {
    warn(`cannot read teams dir ${TEAMS_DIR}: ${e.message}`);
  }

  // Scan shared skills pool
  try {
    skills = fs.readdirSync(SHARED_SKILLS_DIR)
      .filter(d => fs.existsSync(path.join(SHARED_SKILLS_DIR, d, 'SKILL.md')));
  } catch (e) {
    warn(`cannot read shared skills dir ${SHARED_SKILLS_DIR}: ${e.message}`);
  }
  return { agents, skills };
};

/**
 * Resolve team directory and validate it exists. Rejects unsafe names.
 */
const resolveTeam = (teamName) => {
  if (!isValidTeamName(teamName)) {
    warn(`invalid team name "${teamName}" — expected [A-Za-z0-9][A-Za-z0-9_.-]{0,63}`);
    return null;
  }
  const teamDir = path.join(TEAMS_DIR, teamName);
  // Defense-in-depth: ensure resolved path stays under TEAMS_DIR.
  const resolved = path.resolve(teamDir);
  const teamsRoot = path.resolve(TEAMS_DIR) + path.sep;
  if (!(resolved + path.sep).startsWith(teamsRoot)) {
    warn(`team path "${teamName}" escapes teams root`);
    return null;
  }
  if (!fs.existsSync(teamDir)) {
    let available = [];
    try {
      available = fs.readdirSync(TEAMS_DIR).filter(
        d => fs.existsSync(path.join(TEAMS_DIR, d, 'AGENTS.md'))
      );
    } catch { /* ignore */ }
    warn(`Team "${teamName}" not found. Available: ${available.join(', ')}`);
    return null;
  }
  return teamDir;
};

/**
 * Load the marketplace skill map from .claude-plugin/marketplace.json.
 * Builds a lookup table: skills-plugin-name → [skillName, ...]
 * so that team plugins can resolve their skills via the `dependencies` field
 * without duplicating the skills list in every team plugin.json.
 */
const loadMarketplaceSkillMap = () => {
  const marketplacePath = path.join(REPO_ROOT, '.claude-plugin', 'marketplace.json');
  try {
    const marketplace = JSON.parse(fs.readFileSync(marketplacePath, 'utf8'));
    const map = {};
    for (const plugin of (marketplace.plugins || [])) {
      if (plugin.skills && Array.isArray(plugin.skills)) {
        map[plugin.name] = plugin.skills.map(p => path.basename(p));
      }
    }
    return map;
  } catch (e) {
    warn(`Failed to load marketplace.json (${e.message}), skills resolution via dependencies will not work`);
    return {};
  }
};

/**
 * Load team dependency declarations from .claude-plugin/plugin.json.
 * Extracts agent/skill names from the path arrays:
 *   agents: ["./agents/name.md"] → ["name"]
 *   skills: ["./skills/name"]    → ["name"]
 *
 * When plugin.json does not declare `skills` directly (post-cd5b220 architecture),
 * falls back to resolving skills through marketplace.json using the `dependencies`
 * field (e.g. "ops-direct-invoke-skills" → [ascendc-api-best-practices, ...]).
 *
 * Returns { agents: string[], skills: string[] } or null on failure.
 */
const loadTeamDeps = (teamDir, skillMap) => {
  const manifestPath = path.join(teamDir, '.claude-plugin', 'plugin.json');
  try {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const agents = (manifest.agents || []).map(p => path.basename(p, '.md'));
    let skills = (manifest.skills || []).map(p => path.basename(p));
    // Fallback: resolve skills from marketplace.json via dependencies
    if (skills.length === 0 && Array.isArray(manifest.dependencies) && skillMap) {
      for (const dep of manifest.dependencies) {
        const depSkills = skillMap[dep];
        if (depSkills) {
          skills = skills.concat(depSkills);
        }
      }
    }
    return { agents, skills };
  } catch (e) {
    warn(`Failed to load ${manifestPath} (${e.message}), no agents/skills will be installed`);
    return null;
  }
};

/**
 * Parse YAML frontmatter from markdown content.
 * Returns { data: object, content: string }.
 *
 * Supports: scalar values, simple arrays (- item), and one-level nested
 * objects (indented key: value under a parent key with empty value).
 * Tolerates CRLF line endings.
 */
const parseFrontmatter = (content) => {
  const normalized = content.replace(/\r\n/g, '\n');
  const match = normalized.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) return { data: {}, content };

  const yamlBlock = match[1];
  const body = match[2];
  const data = {};

  let currentKey = null;
  let currentType = null; // 'array' | 'object' | null

  for (const line of yamlBlock.split('\n')) {
    // Top-level key: value
    const kvMatch = line.match(/^(\w[\w-]*):\s*(.*)$/);
    if (kvMatch) {
      const [, key, value] = kvMatch;
      if (value.trim() === '') {
        currentKey = key;
        currentType = null;
        data[key] = undefined;
      } else {
        currentKey = null;
        currentType = null;
        data[key] = value.trim();
      }
      continue;
    }

    if (!currentKey) continue;

    const arrayMatch = line.match(/^\s+-\s+(.+)$/);
    if (arrayMatch) {
      if (currentType === null) {
        data[currentKey] = [];
        currentType = 'array';
      }
      if (currentType === 'array') {
        data[currentKey].push(arrayMatch[1].trim());
      }
      continue;
    }

    const nestedMatch = line.match(/^\s+([\w][\w-]*):\s+(.+)$/);
    if (nestedMatch) {
      if (currentType === null) {
        data[currentKey] = {};
        currentType = 'object';
      }
      if (currentType === 'object') {
        data[currentKey][nestedMatch[1]] = nestedMatch[2].trim();
      }
      continue;
    }
  }

  for (const key of Object.keys(data)) {
    if (data[key] === undefined) data[key] = {};
  }

  return { data, content: body };
};

/**
 * Strip YAML frontmatter (--- ... ---) from markdown content.
 * Tolerates CRLF line endings.
 */
const stripFrontmatter = (content) => {
  const normalized = content.replace(/\r\n/g, '\n');
  const match = normalized.match(/^---\n[\s\S]*?\n---\n?([\s\S]*)$/);
  return match ? match[1] : content;
};

/**
 * Coerce YAML scalar to boolean: true/false strings or native booleans.
 * Returns undefined if the value is absent.
 */
const coerceBool = (v) => {
  if (v === undefined || v === null) return undefined;
  if (typeof v === 'boolean') return v;
  if (typeof v === 'string') {
    const s = v.trim().toLowerCase();
    if (s === 'true') return true;
    if (s === 'false') return false;
  }
  return undefined;
};

/**
 * Coerce YAML scalar to a finite number, or undefined if invalid.
 */
const coerceNumber = (v) => {
  if (v === undefined || v === null || v === '') return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
};

/**
 * Parse an agent .md file into a config entry compatible with OpenCode's
 * Config.Agent schema: { name, description, mode, prompt, permission, ... }
 */
const parseAgentMd = (filePath) => {
  let raw;
  try { raw = fs.readFileSync(filePath, 'utf8'); }
  catch (e) { warn(`cannot read agent file ${filePath}: ${e.message}`); return null; }

  let data, content;
  try { ({ data, content } = parseFrontmatter(raw)); }
  catch (e) { warn(`failed to parse frontmatter in ${filePath}: ${e.message}`); return null; }

  if (!data.name) {
    warn(`agent file ${filePath} missing required "name" field`);
    return null;
  }

  const agent = { prompt: (content || '').trim() };
  if (data.description) agent.description = data.description;
  if (data.mode) agent.mode = data.mode;
  if (data.model) agent.model = data.model;

  const hiddenVal = coerceBool(data.hidden);
  if (hiddenVal !== undefined) agent.hidden = hiddenVal;

  const tempVal = coerceNumber(data.temperature);
  if (tempVal !== undefined) agent.temperature = tempVal;

  const stepsVal = coerceNumber(data.steps);
  if (stepsVal !== undefined) agent.steps = stepsVal;

  if (data.permission && typeof data.permission === 'object' && !Array.isArray(data.permission)) {
    agent.permission = data.permission;
  }

  return { name: data.name, config: agent };
};

/**
 * Build agent config entries for all declared team agents.
 * Returns a map of { agentName: agentConfig } ready to merge into config.agent.
 *
 * Resolution order:
 *   1. teamDir/agents/ (specific team, post-cd5b220 architecture)
 *   2. SHARED_AGENTS_DIR/ (legacy shared pool)
 *   3. Scan all team directories (all mode, when teamDir is null)
 */
const buildAgentConfigs = (depAgents, teamDir) => {
  const agents = {};
  for (const name of depAgents) {
    let src = null;

    // 1. Specific team directory
    if (teamDir) {
      const candidate = path.join(teamDir, 'agents', `${name}.md`);
      if (fs.existsSync(candidate)) {
        src = candidate;
      }
    }

    // 2. Shared pool (legacy)
    if (!src) {
      const candidate = path.join(SHARED_AGENTS_DIR, `${name}.md`);
      if (fs.existsSync(candidate)) {
        src = candidate;
      }
    }

    // 3. All-mode fallback: scan every team directory
    if (!src && !teamDir) {
      try {
        const teams = fs.readdirSync(TEAMS_DIR);
        for (const t of teams) {
          const candidate = path.join(TEAMS_DIR, t, 'agents', `${name}.md`);
          if (fs.existsSync(candidate)) {
            src = candidate;
            break;
          }
        }
      } catch { /* ignore */ }
    }

    if (!src) {
      warn(`agent file for "${name}" not found in any team or shared pool`);
      continue;
    }

    const parsed = parseAgentMd(src);
    if (parsed) {
      agents[parsed.name] = parsed.config;
    }
  }
  return agents;
};

/**
 * Async sleep helper.
 */
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/**
 * Clone a git repository into `dstDir` using execFile (no shell), asynchronously.
 * Uses --depth=1 --single-branch for a shallow, fast clone. Cleans up partial
 * checkouts, serializes concurrent callers with a lock file, and resolves to
 * `true` on success.
 */
const LOCK_STALE_MS = 10 * 60 * 1000;

const isPidAlive = (pid) => {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; }
  catch (e) { return e.code === 'EPERM'; }
};

const isStaleLock = (lockFile) => {
  let stat;
  try { stat = fs.statSync(lockFile); } catch { return false; }
  if (Date.now() - stat.mtimeMs > LOCK_STALE_MS) return true;
  try {
    const pid = parseInt(fs.readFileSync(lockFile, 'utf8').trim(), 10);
    if (Number.isInteger(pid) && !isPidAlive(pid)) return true;
  } catch { /* unreadable → treat as live to be safe */ }
  return false;
};

const cloneRepo = async (url, dstDir) => {
  // Fast path: already a usable clone.
  if (isUsableClone(dstDir)) return true;

  // Ensure parent dir exists so the lock-file create + git clone can proceed.
  // First run after a fresh install has no cache tree yet.
  try { fs.mkdirSync(path.dirname(dstDir), { recursive: true }); }
  catch (e) { warn(`cannot create parent dir for ${dstDir}: ${e.message}`); return false; }

  // Partial clone (e.g. interrupted by previous run) — blow it away and retry.
  if (fs.existsSync(dstDir)) {
    warn(`detected incomplete clone at ${dstDir}, removing before retry`);
    rmRfQuiet(dstDir);
  }

  // Best-effort exclusive-create lock so two parallel sessions don't race.
  // If a previous crash left a stale lock (dead pid or older than 10 min),
  // reclaim it instead of waiting forever.
  const lockFile = `${dstDir}.lock`;
  let lockFd = null;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      lockFd = fs.openSync(lockFile, 'wx');
      try { fs.writeSync(lockFd, String(process.pid)); } catch { /* ignore */ }
      break;
    } catch (e) {
      if (e.code !== 'EEXIST') {
        warn(`cannot create lock ${lockFile}: ${e.message}`);
        break;
      }
      if (attempt === 0 && isStaleLock(lockFile)) {
        try { fs.unlinkSync(lockFile); continue; }
        catch { /* lost the race, fall through to polling */ }
      }
      // Another live process is cloning; poll asynchronously.
      for (let i = 0; i < 60; i++) {
        if (isUsableClone(dstDir)) return true;
        await sleep(500);
      }
      return isUsableClone(dstDir);
    }
  }

  try {
    await execFileAsync('git', [
      'clone', '--quiet',
      '--depth', '1',
      '--single-branch',
      url, dstDir,
    ], { timeout: 120000 });
  } catch (e) {
    warn(`git clone ${url} failed: ${e.message}`);
    rmRfQuiet(dstDir);
    return false;
  } finally {
    if (lockFd !== null) {
      try { fs.closeSync(lockFd); } catch { /* ignore */ }
      try { fs.unlinkSync(lockFile); } catch { /* ignore */ }
    }
  }

  return isUsableClone(dstDir);
};

/**
 * Lazily ensure asc-devkit is cloned and cleaned.
 * Only applicable to teams that use asc-devkit (e.g., ops-direct-invoke).
 * Resolves to the devkit dir on success, null on failure.
 *
 * Cleanup (clean_markdown.py) is idempotent-gated via a version marker
 * (`.cannbot-cleaned-vN`) inside the devkit. If the marker is missing —
 * either because it's a fresh clone, a half-cleaned tree from a previous
 * aborted run, or a clone that pre-dates the current cleaner version — we
 * re-run cleanup. If Python3 is missing the clone is still returned (the
 * raw tree is usable, just with un-stripped markdown noise).
 */
const CLEAN_MARKER = '.cannbot-cleaned-v1';
const ensureAscDevkit = async (teamName) => {
  const devkitDir = path.join(teamCacheDir(teamName), 'asc-devkit');
  if (!await cloneRepo('https://gitcode.com/cann/asc-devkit.git', devkitDir)) {
    return null;
  }

  const markerPath = path.join(devkitDir, CLEAN_MARKER);
  if (fs.existsSync(markerPath)) return devkitDir;

  const cleanScript = path.join(SHARED_SKILLS_DIR, 'ascendc-docs-search/scripts/clean_markdown.py');
  if (fs.existsSync(cleanScript)) {
    try {
      await execFileAsync('python3', [cleanScript, '--dir', devkitDir, '--no-backup', '--quiet'],
        { timeout: 60000 });
      try { fs.writeFileSync(markerPath, new Date().toISOString()); }
      catch (e) { warn(`cannot write clean marker ${markerPath}: ${e.message}`); }
    } catch (e) {
      warn(`clean_markdown.py failed (non-fatal): ${e.message}`);
    }
  }

  return devkitDir;
};

/**
 * Lazily ensure PyPTO source repo is cloned.
 * Only applicable to teams that use PyPTO (e.g., pypto-op-orchestrator).
 * Resolves to the pypto dir on success, null on failure.
 */
const ensurePyptoRepo = async (teamName) => {
  const pyptoDir = path.join(teamCacheDir(teamName), 'pypto');
  if (!await cloneRepo('https://gitcode.com/cann/pypto.git', pyptoDir)) {
    return null;
  }
  return pyptoDir;
};

/**
 * Build the bootstrap context string for a given team.
 * Mirrors the logic in hooks/session-start (bash version).
 */
const buildBootstrapContent = (teamDir, teamName, depAgents) => {
  const agentsMdPath = path.join(teamDir, 'AGENTS.md');
  if (!fs.existsSync(agentsMdPath)) return null;

  let raw;
  try { raw = fs.readFileSync(agentsMdPath, 'utf8'); }
  catch (e) { warn(`cannot read ${agentsMdPath}: ${e.message}`); return null; }

  const agentsContent = stripFrontmatter(raw);

  let pathGuide = `## CANNBot Installation Paths (Team: ${teamName})\n`;
  let orchestratorIdentity;

  if (teamName === 'pypto-op-orchestrator') {
    orchestratorIdentity = 'PyPTO Operator Development Orchestrator';
    const pyptoDir = path.join(teamCacheDir(teamName), 'pypto');
    const agentsDir = path.join(teamDir, 'agents');

    pathGuide += `
All relative references in this document resolve to absolute paths:
- \`pypto/\` → \`${pyptoDir}/\`
- \`agents/\` → \`${agentsDir}/\`

Paths starting with \`custom/{op}/\` are relative to the user's current working directory.`;
  } else {
    orchestratorIdentity = 'Ascend C Kernel Development Orchestrator';
    const workflowsDir = path.join(teamDir, 'workflows');
    const ascDevkitDir = path.join(teamCacheDir(teamName), 'asc-devkit');

    if (fs.existsSync(workflowsDir)) {
      pathGuide += `
All \`workflows/\` references in this document resolve to absolute paths:
- \`workflows/task-prompts.md\` → \`${teamDir}/workflows/task-prompts.md\`
- \`workflows/development-guide.md\` → \`${teamDir}/workflows/development-guide.md\`
- \`workflows/scripts/init_operator_project.sh\` → \`${teamDir}/workflows/scripts/init_operator_project.sh\`
- \`workflows/scripts/verify_environment.sh\` → \`${teamDir}/workflows/scripts/verify_environment.sh\`
- \`workflows/templates/\` → \`${teamDir}/workflows/templates/\``;
    }

    if (fs.existsSync(ascDevkitDir) || teamName === 'ops-direct-invoke') {
      pathGuide += `\n- \`asc-devkit/\` → \`${ascDevkitDir}/\``;
    }

    pathGuide += `\n\nPaths starting with \`ops/{operator_name}/\` are relative to the user's current working directory.`;
  }

  const agentMentions = depAgents.map(a => `\`@${a}\``).join(', ');
  const toolMapping = `## Tool Mapping for OpenCode

When AGENTS.md references Claude Code tools, substitute OpenCode equivalents:
- Subagent dispatch → Use OpenCode's \`@mention\` system (${agentMentions || 'see agents/ directory'})
- \`Read\`, \`Write\`, \`Edit\`, \`Bash\` → Your native tools
- \`skill\` tool → OpenCode's native \`skill\` tool`;

  // Tell the LLM how to translate the relative paths in skill docs to
  // shell-friendly env vars vs. tool-friendly absolute paths.
  let envGuide = '';
  if (teamName === 'pypto-op-orchestrator') {
    envGuide = `## Environment Variables (auto-injected into every shell call)

- \`$PYPTO_DIR\` → ${path.join(teamCacheDir(teamName), 'pypto')} (always set; may point to a path that is still being cloned on first use)

When you see relative paths like \`pypto/...\` in the skill docs:
- **In Bash tool commands**: prefer \`"$PYPTO_DIR"/...\` over \`pypto/...\`. The CWD has no \`pypto/\` symlink.
- **In Read/Write/Edit tool file paths**: use the absolute path from the path map above (tools do not expand env vars).`;
  } else {
    envGuide = `## Environment Variables (auto-injected into every shell call)

- \`$ASC_DEVKIT_DIR\` → ${path.join(teamCacheDir(teamName), 'asc-devkit')} (always set; may point to a path that is still being cloned on first use)

When you see relative paths like \`asc-devkit/docs/api/context/...\` in the skill docs:
- **In Bash tool commands**: prefer \`"$ASC_DEVKIT_DIR"/docs/api/context/...\` over \`asc-devkit/...\`. The CWD has no \`asc-devkit/\` symlink — the env var is the only way to reach the devkit from shell.
- **In Read/Write/Edit tool file paths**: use the absolute path from the path map above (tools do not expand env vars).

If a command fails because the path doesn't exist yet, \`$CANNBOT_REPO_PENDING\` will be set to the repo name — wait a few seconds and retry.`;
  }

  return `${CONTEXT_TAG_OPEN}
You are CANNBot - ${orchestratorIdentity}.
Active team: ${teamName}

${pathGuide}

${envGuide}

${toolMapping}

${agentsContent}
</${CONTEXT_TAG}>`;
};

/**
 * Resolve the absolute directories for the team's declared skills.
 * OpenCode's `config.skills.paths` accepts absolute paths and walks each
 * one looking for `**\/SKILL.md` — so we can point directly at the plugin
 * cache instead of writing symlinks into the user's project dir.
 *
 * Searches team-specific skills/ first (post-cd5b220 self-contained architecture),
 * then falls back to the shared pool (legacy architecture).
 *
 * Returns an array of absolute paths (one per valid skill).
 */
const resolveSkillPaths = (depSkills, teamDir) => {
  const out = [];
  for (const name of depSkills) {
    const teamSrc = teamDir ? path.join(teamDir, 'skills', name) : null;
    const sharedSrc = path.join(SHARED_SKILLS_DIR, name);
    const src = teamSrc && fs.existsSync(path.join(teamSrc, 'SKILL.md')) ? teamSrc : sharedSrc;
    if (!fs.existsSync(path.join(src, 'SKILL.md'))) {
      warn(`Declared skill "${name}" not found in ${SHARED_SKILLS_DIR} or team skills dir`);
      continue;
    }
    out.push(src);
  }
  return out;
};

/**
 * Best-effort cleanup: if a previous version of the plugin created
 * `.opencode/agents/` or `.opencode/skills/` symlinks inside the cwd,
 * remove the ones that still point into the shared pool. Leaves real
 * files and user-authored symlinks (pointing elsewhere) alone.
 *
 * Gated on a marker file `.opencode/.cannbot-legacy-v0`: only runs in
 * project dirs that were actually touched by the legacy plugin version,
 * never in arbitrary cwds that happen to have their own `.opencode/`.
 */
const LEGACY_MARKER = '.cannbot-legacy-v0';
const cleanupLegacyProjectLinks = (projectDir) => {
  const markerPath = path.join(projectDir, '.opencode', LEGACY_MARKER);
  if (!fs.existsSync(markerPath)) return;

  const agentsDir = path.join(projectDir, '.opencode', 'agents');
  const skillsDir = path.join(projectDir, '.opencode', 'skills');

  const sweep = (dir, allowedPrefix) => {
    let entries;
    try { entries = fs.readdirSync(dir); } catch { return false; }
    let removed = 0;
    for (const e of entries) {
      const before = fs.existsSync(path.join(dir, e));
      safeUnlinkStaleSymlink(path.join(dir, e), allowedPrefix);
      if (before && !fs.existsSync(path.join(dir, e))) removed++;
    }
    // Remove the directory itself if it is now empty (and itself a real dir).
    try {
      if (fs.readdirSync(dir).length === 0) fs.rmdirSync(dir);
    } catch { /* ignore */ }
    return removed > 0;
  };

  sweep(agentsDir, SHARED_AGENTS_DIR);
  sweep(skillsDir, SHARED_SKILLS_DIR);

  // Marker has served its purpose; remove it so we never sweep again.
  try { fs.unlinkSync(markerPath); } catch { /* ignore */ }

  // Remove parent .opencode dir only if we cleaned both subdirs and it's empty.
  try {
    const parent = path.join(projectDir, '.opencode');
    if (fs.readdirSync(parent).length === 0) fs.rmdirSync(parent);
  } catch { /* ignore */ }
};

export const CANNBotPlugin = async ({ client, directory }, options) => {
  // Only activate when explicitly declared in an opencode.json — otherwise
  // OpenCode's auto-discovery of .opencode/plugins/ would activate this file
  // anywhere it appears in a git tree (e.g. the source repo itself).
  if (!isPluginDeclared(directory)) return {};

  const teamName = options && options.team;
  const isAllTeams = teamName === 'all';

  // Validate team name if provided (and not the "all" literal).
  if (teamName && !isAllTeams && !isValidTeamName(teamName)) {
    warn(`ignoring invalid team name "${teamName}"`);
    return {};
  }

  // Default team (no team specified) → ops-direct-invoke
  const effectiveTeam = teamName ? (isAllTeams ? null : teamName) : DEFAULT_TEAM;
  const teamDir = effectiveTeam ? resolveTeam(effectiveTeam) : null;

  // Team specified but not found: abort
  if (effectiveTeam && !teamDir) return {};

  // "all" → install everything; specific team → install only its deps
  const skillMap = loadMarketplaceSkillMap();
  const deps = isAllTeams
    ? loadAllDeps()
    : (teamDir ? (loadTeamDeps(teamDir, skillMap) || { agents: [], skills: [] }) : loadAllDeps());

  // Resolve skill paths directly from the plugin cache. OpenCode's
  // config.skills.paths takes absolute paths, so there is no need to
  // create `.opencode/skills/` symlinks inside the user's cwd.
  const skillPaths = resolveSkillPaths(deps.skills, teamDir);

  // Pre-parse agent configs for injection via config hook. Agents are
  // registered entirely via `config.agent` — no `.opencode/agents/` symlinks.
  const agentConfigs = buildAgentConfigs(deps.agents, teamDir);

  // One-shot: remove `.opencode/agents/` and `.opencode/skills/` symlink
  // directories that earlier versions of this plugin left behind in random
  // cwds. Only symlinks pointing into the shared pool are touched.
  cleanupLegacyProjectLinks(directory);

  // For bootstrap context injection, fall back to default team when no team or "all" specified
  const contextTeamDir = teamDir || resolveTeam(DEFAULT_TEAM);
  const contextTeamName = effectiveTeam || DEFAULT_TEAM;

  // External repos (asc-devkit / pypto) clone policy:
  //   - At plugin init we *eagerly* kick off the clone in the background
  //     (fire-and-forget, no await), so a fresh install warms up while the
  //     user is typing. Init itself stays ~instant — the promise is not
  //     awaited here.
  //   - shell.env advertises the absolute paths as env vars unconditionally
  //     (once the team is known), even before the clone finishes. This means
  //     the LLM never sees an empty $ASC_DEVKIT_DIR; at worst it sees a
  //     clear "No such file" once, then the second call works.
  //   - shell.env also re-awaits the in-flight clone briefly when the
  //     command actually references the repo, to let the first such call
  //     succeed whenever the clone finishes within a few seconds.
  let _lazyClonePromise = null;
  let _lazyCloneFailedUntil = 0;
  const CLONE_FAILURE_COOLDOWN_MS = 60_000;
  const ensureTeamRepoLazy = () => {
    if (!contextTeamDir) return null;
    if (_lazyClonePromise) return _lazyClonePromise;
    // Back off after a failure so offline / firewalled users don't eat the
    // COLD_CLONE_WAIT_MS penalty on *every* command that references the repo.
    if (Date.now() < _lazyCloneFailedUntil) return null;
    const ensure = contextTeamName === 'pypto-op-orchestrator'
      ? ensurePyptoRepo
      : ensureAscDevkit;
    _lazyClonePromise = (async () => {
      try { return Boolean(await ensure(contextTeamName)); }
      catch (e) { warn(`lazy repo setup failed: ${e.message}`); return false; }
    })();
    // Allow a retry after the cooldown window: clear the cached promise when
    // it resolves to false (clone didn't land) and stamp a timestamp so we
    // skip re-trying for CLONE_FAILURE_COOLDOWN_MS.
    _lazyClonePromise.then(
      (ok) => {
        if (!ok) {
          _lazyCloneFailedUntil = Date.now() + CLONE_FAILURE_COOLDOWN_MS;
          _lazyClonePromise = null;
        }
      },
      () => {
        _lazyCloneFailedUntil = Date.now() + CLONE_FAILURE_COOLDOWN_MS;
        _lazyClonePromise = null;
      },
    );
    return _lazyClonePromise;
  };

  // Eager warm-up: start the clone now if the cache is cold, OR if the
  // devkit is present but the cleanup marker is missing (partially cleaned
  // tree, or a clone made by an older plugin version). Fire-and-forget.
  if (contextTeamDir) {
    const devkitDir = path.join(teamCacheDir(contextTeamName), 'asc-devkit');
    const pyptoDir = path.join(teamCacheDir(contextTeamName), 'pypto');
    const needsDevkit = contextTeamName !== 'pypto-op-orchestrator' && (
      !isUsableClone(devkitDir) || !fs.existsSync(path.join(devkitDir, CLEAN_MARKER))
    );
    const needsPypto = contextTeamName === 'pypto-op-orchestrator' && !isUsableClone(pyptoDir);
    if (needsDevkit || needsPypto) ensureTeamRepoLazy();
  }

  return {
    // Register skills directory and inject agent configs for discovery.
    config: async (config) => {
      config.skills = config.skills || {};
      config.skills.paths = config.skills.paths || [];
      for (const p of skillPaths) {
        if (!config.skills.paths.includes(p)) config.skills.paths.push(p);
      }

      config.agent = config.agent || {};
      for (const [name, agentCfg] of Object.entries(agentConfigs)) {
        if (!config.agent[name]) {
          config.agent[name] = agentCfg;
        }
      }
    },

    // Inject AGENTS.md orchestration context into the first user message.
    // NON-BLOCKING: does NOT await the background clone — the bootstrap
    // context is pure string construction, independent of the external repo.
    'experimental.chat.messages.transform': async (_input, output) => {
      if (!contextTeamDir) return;

      const bootstrap = buildBootstrapContent(contextTeamDir, contextTeamName, deps.agents);
      if (!bootstrap || !output.messages?.length) return;

      const firstUser = output.messages.find(m => m.info && m.info.role === 'user');
      if (!firstUser || !firstUser.parts || !firstUser.parts.length) return;

      // Only inject once — match the opening tag strictly so stray mentions
      // of the bare identifier elsewhere in the message don't block injection.
      if (firstUser.parts.some(p => p.type === 'text' && typeof p.text === 'string' && p.text.includes(CONTEXT_TAG_OPEN))) return;

      // Construct a clean text part. If the first existing part happens to be
      // text we preserve its incidental fields (id, etc.); otherwise we do
      // NOT inherit fields from non-text parts (file/image) which would
      // confuse downstream consumers.
      const ref = firstUser.parts[0];
      const basePart = ref && ref.type === 'text' ? ref : {};
      firstUser.parts.unshift({ ...basePart, type: 'text', text: bootstrap });
    },

    // Expose cache-absolute paths to every shell invocation (Bash tool,
    // command bash blocks, PTY) so scripts and the LLM can locate the
    // devkit / pypto repos without symlinks in the user's cwd.
    //
    // Env var policy: once the team is known we ALWAYS set the path vars,
    // even if the clone hasn't finished yet. The LLM and scripts see the
    // eventual absolute path; commands issued before clone completes fail
    // with a clear "No such file" (far better than an empty-variable
    // `/docs/...: No such file`), and the next command naturally succeeds.
    //
    // For commands that actually reference the repo, we briefly await the
    // in-flight clone (≤ COLD_CLONE_WAIT_MS) to let the first such call
    // succeed when the clone finishes in time. CANNBOT_REPO_PENDING is
    // exported so downstream tooling can distinguish "still cloning" from
    // "genuinely missing".
    'shell.env': async (input, output) => {
      if (!contextTeamDir) return;
      const devkitDir = path.join(teamCacheDir(contextTeamName), 'asc-devkit');
      const pyptoDir = path.join(teamCacheDir(contextTeamName), 'pypto');

      const isPyptoTeam = contextTeamName === 'pypto-op-orchestrator';

      // Always advertise the target path for the active team, clone-ready
      // or not. Non-pypto teams also expose ASC_DEVKIT_DIR since that is
      // the primary devkit the skill docs reference.
      output.env = output.env || {};
      if (!isPyptoTeam) output.env.ASC_DEVKIT_DIR = devkitDir;
      if (isPyptoTeam) output.env.PYPTO_DIR = pyptoDir;

      const devkitReady = isUsableClone(devkitDir);
      const pyptoReady = isUsableClone(pyptoDir);
      if (isPyptoTeam ? pyptoReady : devkitReady) return;

      const cmd = typeof input?.command === 'string' ? input.command : '';
      if (!cmd) return;

      // Only wait on clone when the command *uses* the repo — references
      // the env var or a subpath prefix (e.g. `asc-devkit/docs/...`). Bare
      // word matches like `grep pypto README.md` do not count.
      const wantsDevkit = !devkitReady && /\$ASC_DEVKIT_DIR\b|\basc[-_]?devkit\//i.test(cmd);
      const wantsPypto = !pyptoReady && isPyptoTeam
        && /\$PYPTO_DIR\b|\bpypto\/[A-Za-z0-9_.-]/i.test(cmd);

      if (!wantsDevkit && !wantsPypto) return;

      const COLD_CLONE_WAIT_MS = 3000;
      const pending = ensureTeamRepoLazy();
      if (pending) await Promise.race([pending, sleep(COLD_CLONE_WAIT_MS)]);

      const stillWantsDevkit = wantsDevkit && !isUsableClone(devkitDir);
      const stillWantsPypto = wantsPypto && !isUsableClone(pyptoDir);
      if (stillWantsDevkit || stillWantsPypto) {
        const which = stillWantsDevkit ? 'asc-devkit' : 'pypto';
        output.env.CANNBOT_REPO_PENDING = which;
        const hint = pending
          ? 'still cloning; this command may fail — retry in a few seconds'
          : 'last clone failed; backing off for up to 60s before retry';
        warn(`${which} ${hint}`);
      }
    }
  };
};
