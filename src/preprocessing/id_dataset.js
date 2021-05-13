const fs = require('fs');

const files = process.argv.slice(2);

console.log('input files: ', files);
let index = 0;

for (const file of files) {
	const jsonData = require(file);
	console.log(file)
	
	const arr = []

	for (const issue of jsonData)
	{
		issue['id'] = index++;
		arr.push(issue)
	}
	const JSON_SPACES = 4;
	const data = JSON.stringify(arr, null, JSON_SPACES)

	fs.writeFileSync(file, data);
}

