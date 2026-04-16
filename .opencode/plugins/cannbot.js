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
 * Each team declares its dependencies in .claude-plugin/plugin.json:
 *   { "deps": { "agents": [...], "skills": [...] } }
 * When team is "all", all agents and skills from the shared pool are installed.
 */

import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../..');
const TEAMS_DIR = path.join(REPO_ROOT, 'ops/teams');
const SHARED_AGENTS_DIR = path.join(REPO_ROOT, 'ops/agents');
const SHARED_SKILLS_DIR = path.join(REPO_ROOT, 'ops/skills');

const DEFAULT_TEAM = 'ops-direct-invoke';
const CONTEXT_TAG = 'CANNBOT_CONTEXT';

/**
 * Load all agents/skills from the shared pool (no filtering).
 * Returns { agents: string[], skills: string[] }.
 */
const loadAllDeps = () => {
  let agents = [];
  let skills = [];
  try {
    agents = fs.readdirSync(SHARED_AGENTS_DIR)
      .filter(f => f.endsWith('.md'))
      .map(f => f.replace(/\.md$/, ''));
  } catch { /* empty */ }
  try {
    skills = fs.readdirSync(SHARED_SKILLS_DIR)
      .filter(d => fs.existsSync(path.join(SHARED_SKILLS_DIR, d, 'SKILL.md')));
  } catch { /* empty */ }
  return { agents, skills };
};

/**
 * Resolve team directory and validate it exists.
 */
const resolveTeam = (teamName) => {
  const teamDir = path.join(TEAMS_DIR, teamName);
  if (!fs.existsSync(teamDir)) {
    const available = fs.readdirSync(TEAMS_DIR).filter(
      d => fs.existsSync(path.join(TEAMS_DIR, d, 'AGENTS.md'))
    );
    console.error(
      `[CANNBot] Team "${teamName}" not found. Available: ${available.join(', ')}`
    );
    return null;
  }
  return teamDir;
};

/**
 * Load team dependency declarations from .claude-plugin/plugin.json.
 * Extracts agent/skill names from the path arrays:
 *   agents: ["./agents/name.md"] → ["name"]
 *   skills: ["./skills/name"]    → ["name"]
 * Returns { agents: string[], skills: string[] } or null on failure.
 */
const loadTeamDeps = (teamDir) => {
  const manifestPath = path.join(teamDir, '.claude-plugin', 'plugin.json');
  try {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const agents = (manifest.agents || []).map(
      p => path.basename(p, '.md')
    );
    const skills = (manifest.skills || []).map(
      p => path.basename(p)
    );
    return { agents, skills };
  } catch {
    console.error(`[CANNBot] Failed to load ${manifestPath}, no agents/skills will be installed`);
    return null;
  }
};

/**
 * Parse YAML frontmatter from markdown content.
 * Returns { data: object, content: string }.
 *
 * Supports: scalar values, simple arrays (- item), and one-level nested
 * objects (indented key: value under a parent key with empty value).
 */
const parseFrontmatter = (content) => {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
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
        // Start of a nested block (array or object) - determined by first child
        currentKey = key;
        currentType = null;
        data[key] = undefined; // placeholder, resolved on first child
      } else {
        currentKey = null;
        currentType = null;
        data[key] = value.trim();
      }
      continue;
    }

    if (!currentKey) continue;

    // Array item:   - value
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

    // Nested object entry:   key: value  (indented)
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

  // Clean up any unresolved placeholders
  for (const key of Object.keys(data)) {
    if (data[key] === undefined) data[key] = {};
  }

  return { data, content: body };
};

/**
 * Strip YAML frontmatter (--- ... ---) from markdown content.
 */
const stripFrontmatter = (content) => {
  const match = content.match(/^---\n[\s\S]*?\n---\n([\s\S]*)$/);
  return match ? match[1] : content;
};

/**
 * Parse an agent .md file into a config entry compatible with OpenCode's
 * Config.Agent schema: { name, description, mode, prompt, permission, ... }
 */
const parseAgentMd = (filePath) => {
  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    const { data, content } = parseFrontmatter(raw);
    if (!data.name) return null;

    const agent = {
      prompt: content.trim(),
    };
    if (data.description) agent.description = data.description;
    if (data.mode) agent.mode = data.mode;
    if (data.model) agent.model = data.model;
    if (data.hidden) agent.hidden = data.hidden === 'true' || data.hidden === true;
    if (data.temperature) agent.temperature = Number(data.temperature);
    if (data.steps) agent.steps = Number(data.steps);

    // Parse permission block
    if (data.permission && typeof data.permission === 'object' && !Array.isArray(data.permission)) {
      agent.permission = data.permission;
    }

    return { name: data.name, config: agent };
  } catch {
    return null;
  }
};

/**
 * Build agent config entries for all declared team agents.
 * Returns a map of { agentName: agentConfig } ready to merge into config.agent.
 */
const buildAgentConfigs = (depAgents) => {
  const agents = {};
  for (const name of depAgents) {
    const src = path.join(SHARED_AGENTS_DIR, `${name}.md`);
    const parsed = parseAgentMd(src);
    if (parsed) {
      agents[parsed.name] = parsed.config;
    }
  }
  return agents;
};

/**
 * Lazily ensure asc-devkit is cloned and cleaned.
 * Only applicable to teams that use asc-devkit (e.g., ops-direct-invoke).
 */
const ensureAscDevkit = (teamDir) => {
  const devkitDir = path.join(teamDir, 'asc-devkit');
  if (fs.existsSync(devkitDir)) return devkitDir;

  try {
    execSync(
      `git clone --quiet https://gitcode.com/cann/asc-devkit.git "${devkitDir}"`,
      { stdio: 'ignore', timeout: 120000 }
    );
  } catch {
    return devkitDir;
  }

  const cleanScript = path.join(SHARED_SKILLS_DIR, 'ascendc-docs-search/scripts/clean_markdown.py');
  if (fs.existsSync(devkitDir) && fs.existsSync(cleanScript)) {
    try {
      execSync(
        `python3 "${cleanScript}" --dir "${devkitDir}" --no-backup --quiet`,
        { stdio: 'ignore', timeout: 60000 }
      );
    } catch {
      // cleanup failed, non-fatal
    }
  }

  return devkitDir;
};

/**
 * Lazily ensure PyPTO source repo is cloned.
 * Only applicable to teams that use PyPTO (e.g., pypto-op-orchestrator).
 */
const ensurePyptoRepo = (teamDir) => {
  const pyptoDir = path.join(teamDir, 'pypto');
  if (fs.existsSync(pyptoDir)) return pyptoDir;

  try {
    execSync(
      `git clone --quiet https://gitcode.com/cann/pypto.git "${pyptoDir}"`,
      { stdio: 'ignore', timeout: 120000 }
    );
  } catch {
    // clone failed, non-fatal
  }

  return pyptoDir;
};

/**
 * Symlink PyPTO source repo from team cache into the project root so that
 * skills referencing relative `pypto/` paths can find it.
 */
const linkPyptoRepo = (teamDir, projectDir) => {
  const src = path.join(teamDir, 'pypto');
  const dst = path.join(projectDir, 'pypto');
  if (!fs.existsSync(src)) return;
  try {
    if (fs.lstatSync(dst).isSymbolicLink() && fs.readlinkSync(dst) === src) return;
    fs.unlinkSync(dst);
  } catch {
    // dst doesn't exist, proceed
  }
  fs.symlinkSync(src, dst);
};

/**
 * Build the bootstrap context string for a given team.
 * Mirrors the logic in hooks/session-start (bash version).
 */
const buildBootstrapContent = (teamDir, teamName, depAgents) => {
  const agentsMdPath = path.join(teamDir, 'AGENTS.md');
  if (!fs.existsSync(agentsMdPath)) return null;

  const raw = fs.readFileSync(agentsMdPath, 'utf8');
  const agentsContent = stripFrontmatter(raw);

  // Build path guide with absolute paths — team-specific
  let pathGuide = `## CANNBot Installation Paths (Team: ${teamName})\n`;
  let orchestratorIdentity;

  if (teamName === 'pypto-op-orchestrator') {
    // PyPTO team: reference pypto/, skills/, agents/ paths
    orchestratorIdentity = 'PyPTO Operator Development Orchestrator';
    const pyptoDir = path.join(teamDir, 'pypto');
    const skillsDir = path.join(teamDir, 'skills');
    const agentsDir = path.join(teamDir, 'agents');

    pathGuide += `
All relative references in this document resolve to absolute paths:
- \`pypto/\` → \`${pyptoDir}/\`
- \`skills/\` → \`${skillsDir}/\`
- \`agents/\` → \`${agentsDir}/\`

Paths starting with \`custom/{op}/\` are relative to the user's current working directory.`;
  } else {
    // ops-direct-invoke (and default): reference workflows/, asc-devkit/ paths
    orchestratorIdentity = 'Ascend C Kernel Development Orchestrator';
    const workflowsDir = path.join(teamDir, 'workflows');
    const ascDevkitDir = path.join(teamDir, 'asc-devkit');

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

  // Build @mention list from team's declared agents only
  const agentMentions = depAgents.map(a => `\`@${a}\``).join(', ');
  const toolMapping = `## Tool Mapping for OpenCode

When AGENTS.md references Claude Code tools, substitute OpenCode equivalents:
- Subagent dispatch → Use OpenCode's \`@mention\` system (${agentMentions || 'see agents/ directory'})
- \`Read\`, \`Write\`, \`Edit\`, \`Bash\` → Your native tools
- \`skill\` tool → OpenCode's native \`skill\` tool`;

  return `<${CONTEXT_TAG}>
You are CANNBot - ${orchestratorIdentity}.
Active team: ${teamName}

${pathGuide}

${toolMapping}

${agentsContent}
</${CONTEXT_TAG}>`;
};

/**
 * Symlink asc-devkit from team cache into the project root so that
 * verify_environment.sh (which uses the relative path "asc-devkit") can find it.
 */
const linkAscDevkit = (teamDir, projectDir) => {
  const src = path.join(teamDir, 'asc-devkit');
  const dst = path.join(projectDir, 'asc-devkit');
  if (!fs.existsSync(src)) return;
  try {
    if (fs.lstatSync(dst).isSymbolicLink() && fs.readlinkSync(dst) === src) return;
    fs.unlinkSync(dst);
  } catch {
    // dst doesn't exist, proceed
  }
  fs.symlinkSync(src, dst);
};

/**
 * Link only the team's declared skills from the shared ops/skills/ pool
 * into .opencode/skills/ for discovery. OpenCode expects a parent directory
 * containing skill subdirectories (each with SKILL.md), not individual paths.
 *
 * Also cleans up stale symlinks from a previously active team.
 */
const linkSkills = (depSkills, projectDir) => {
  const dstSkillsDir = path.join(projectDir, '.opencode', 'skills');
  fs.mkdirSync(dstSkillsDir, { recursive: true });

  const depSet = new Set(depSkills);

  // Phase 1: link declared skills
  for (const name of depSkills) {
    const src = path.join(SHARED_SKILLS_DIR, name);
    const dst = path.join(dstSkillsDir, name);
    if (!fs.existsSync(src)) {
      console.error(`[CANNBot] Declared skill "${name}" not found in ${SHARED_SKILLS_DIR}`);
      continue;
    }
    try {
      if (fs.lstatSync(dst).isSymbolicLink() && fs.readlinkSync(dst) === src) continue;
      fs.unlinkSync(dst);
    } catch {
      // dst doesn't exist, proceed
    }
    fs.symlinkSync(src, dst);
  }

  // Phase 2: remove stale symlinks that point into SHARED_SKILLS_DIR but are not in depSet
  try {
    for (const entry of fs.readdirSync(dstSkillsDir)) {
      const dst = path.join(dstSkillsDir, entry);
      try {
        if (!fs.lstatSync(dst).isSymbolicLink()) continue;
        const target = fs.readlinkSync(dst);
        if (!target.startsWith(SHARED_SKILLS_DIR)) continue;
        if (!depSet.has(entry)) {
          fs.unlinkSync(dst);
        }
      } catch {
        // skip unreadable entries
      }
    }
  } catch {
    // dstSkillsDir unreadable, skip cleanup
  }

  return dstSkillsDir;
};

/**
 * Link only the team's declared agents from the shared ops/agents/ pool
 * into .opencode/agents/ for @mention discovery.
 *
 * Also cleans up stale symlinks from a previously active team.
 */
const linkAgents = (depAgents, projectDir) => {
  const dstAgentsDir = path.join(projectDir, '.opencode', 'agents');
  fs.mkdirSync(dstAgentsDir, { recursive: true });

  const depSet = new Set(depAgents);

  // Phase 1: link declared agents
  for (const name of depAgents) {
    const src = path.join(SHARED_AGENTS_DIR, `${name}.md`);
    const dst = path.join(dstAgentsDir, `${name}.md`);
    if (!fs.existsSync(src)) {
      console.error(`[CANNBot] Declared agent "${name}" not found in ${SHARED_AGENTS_DIR}`);
      continue;
    }
    try {
      if (fs.lstatSync(dst).isSymbolicLink() && fs.readlinkSync(dst) === src) continue;
      fs.unlinkSync(dst);
    } catch {
      // dst doesn't exist, proceed
    }
    fs.symlinkSync(src, dst);
  }

  // Phase 2: remove stale symlinks that point into SHARED_AGENTS_DIR but are not in depSet
  try {
    for (const file of fs.readdirSync(dstAgentsDir)) {
      const dst = path.join(dstAgentsDir, file);
      try {
        if (!fs.lstatSync(dst).isSymbolicLink()) continue;
        const target = fs.readlinkSync(dst);
        if (!target.startsWith(SHARED_AGENTS_DIR)) continue;
        const agentName = file.replace(/\.md$/, '');
        if (!depSet.has(agentName)) {
          fs.unlinkSync(dst);
        }
      } catch {
        // skip unreadable entries
      }
    }
  } catch {
    // dstAgentsDir unreadable, skip cleanup
  }
};

export const CANNBotPlugin = async ({ client, directory }, options) => {
  const teamName = options && options.team;
  const isAllTeams = teamName === 'all';

  // Default team (no team specified) → ops-direct-invoke
  const effectiveTeam = teamName ? (isAllTeams ? null : teamName) : DEFAULT_TEAM;
  const teamDir = effectiveTeam ? resolveTeam(effectiveTeam) : null;

  // Team specified but not found: abort
  if (effectiveTeam && !teamDir) return {};

  // "all" → install everything; specific team → install only its deps
  const deps = isAllTeams
    ? loadAllDeps()
    : (teamDir ? (loadTeamDeps(teamDir) || { agents: [], skills: [] }) : loadAllDeps());

  let repoReady = false;

  // Link agents and skills into project .opencode/
  linkAgents(deps.agents, directory);
  const linkedSkillsDir = linkSkills(deps.skills, directory);

  // Pre-parse agent configs for injection via config hook
  const agentConfigs = buildAgentConfigs(deps.agents);

  // For bootstrap context injection, fall back to default team when no team or "all" specified
  const contextTeamDir = teamDir || resolveTeam(DEFAULT_TEAM);
  const contextTeamName = effectiveTeam || DEFAULT_TEAM;

  return {
    // Register skills directory and inject agent configs for discovery.
    config: async (config) => {
      config.skills = config.skills || {};
      config.skills.paths = config.skills.paths || [];
      if (fs.existsSync(linkedSkillsDir) && !config.skills.paths.includes(linkedSkillsDir)) {
        config.skills.paths.push(linkedSkillsDir);
      }

      config.agent = config.agent || {};
      for (const [name, agentCfg] of Object.entries(agentConfigs)) {
        if (!config.agent[name]) {
          config.agent[name] = agentCfg;
        }
      }
    },

    // Inject AGENTS.md orchestration context into the first user message.
    'experimental.chat.messages.transform': async (_input, output) => {
      if (!contextTeamDir) return;

      // Lazy repo setup (once per session) — team-specific
      if (!repoReady) {
        repoReady = true;
        const activeTeam = contextTeamName;
        if (activeTeam === 'pypto-op-orchestrator') {
          ensurePyptoRepo(contextTeamDir);
          linkPyptoRepo(contextTeamDir, directory);
        } else {
          ensureAscDevkit(contextTeamDir);
          linkAscDevkit(contextTeamDir, directory);
        }
      }

      const bootstrap = buildBootstrapContent(contextTeamDir, contextTeamName, deps.agents);
      if (!bootstrap || !output.messages.length) return;

      const firstUser = output.messages.find(m => m.info.role === 'user');
      if (!firstUser || !firstUser.parts.length) return;

      // Only inject once
      if (firstUser.parts.some(p => p.type === 'text' && p.text.includes(CONTEXT_TAG))) return;

      const ref = firstUser.parts[0];
      firstUser.parts.unshift({ ...ref, type: 'text', text: bootstrap });
    }
  };
};
