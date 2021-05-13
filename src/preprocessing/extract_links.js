const MarkdownIt = require('markdown-it');
const JSSoup = require('jssoup').default;
const fs = require('fs');
const path = require('path')
md = new MarkdownIt();

const files = process.argv.slice(2);

console.log('input files: ', files);

for (const file of files) {
	const jsonData = require(file);
	const file_name = path.basename(file, '.json')
	console.log(file_name)

	const arr = []

	for (const issue of jsonData)
	{
		let html = md.render(issue['text']);
		//const LINK_REGEX = /<a .*?>(.*?)<\/a>/
		const LINK_REGEX = /<a href="(.*?)">(.*?)<\/a>/g
	
		issue['links']  = [...html.matchAll(LINK_REGEX)].map(x => x[1])
		
		issue['html'] = html.replace(LINK_REGEX, "$2")
		
		const soup = new JSSoup(html)

		for (const img of soup.findAll('img')) {
			img.replaceWith(' IMAGE_TOKEN ')
		}
		issue['html'] = soup.text
		delete issue['text']
		arr.push(issue)
	}
	const JSON_SPACES = 4;
	const data = JSON.stringify(arr, null, JSON_SPACES)

	fs.writeFileSync(`${file_name}_links_extracted.json`, data);
}

