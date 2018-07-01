import lseapprox as lse

values = []

for r in range(0,101):
    if r >= 2:
        A,b,err = lse.lowerapprox(r)
        values.append(err)
        print(str(r), "   ", str(values[r]))
    else:
        values.append(-1)
file = open("table.html","w+")
file.write("<table id=\"myTable\">\n")
file.write("    <tr>\n")
file.write("        <th onclick=\"sortTable(0)\">r</th>\n")
file.write("        <th onclick=\"sortTable(1)\">error</th>\n")
file.write("    </tr>\n")

for r in range(0,len(values)):
    file.write("    <tr>\n")
    file.write("        <td>")
    file.write(str(r))
    file.write("</td>\n")
    file.write("        <td>")
    file.write(str(values[r]))
    file.write("</td>\n")
    file.write("    </tr>\n")
file.write("</table>")
file.close()

