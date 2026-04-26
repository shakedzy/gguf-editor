'use strict';

const { execFileSync } = require('node:child_process');
const path = require('node:path');

exports.default = async function adhocSignMacApp(context) {
  if (context.electronPlatformName !== 'darwin') return;

  const appName = context.packager.appInfo.productFilename;
  const appPath = path.join(context.appOutDir, `${appName}.app`);

  console.log(`  • ad-hoc signing app bundle  app=${appPath}`);
  execFileSync(
    'codesign',
    ['--force', '--deep', '--sign', '-', appPath],
    { stdio: 'inherit' }
  );
};
