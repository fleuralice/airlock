import trio
import httpx
from typing import Literal, Any
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from packaging.markers import Marker
from packaging.requirements import Requirement
import packaging
import collections
import dataclasses
import z3

MarkerExpr = tuple[object, object, object]

@dataclasses.dataclass
class MarkerExpr:
    variable: packaging._parser.Variable
    op: packaging._parser.Op
    value: packaging._parser.Value

@dataclasses.dataclass
class MarkerCombination:
    left: "MarkerCombination | MarkerExpr"
    combination: Literal["or", "and"]
    right: "MarkerCombination | MarkerExpr"

def marker_expr(marker: tuple[object, object, object]):
    assert isinstance(marker, tuple)
    assert marker[1].value not in ("in", "not in", "===", "~=")
    if isinstance(marker[0], packaging._parser.Variable):
        result = MarkerExpr(marker[0], marker[1], marker[2])
    else:
        result = invert_tree(MarkerExpr(marker[2], marker[1], marker[0]))
    assert isinstance(result.variable, packaging._parser.Variable)
    assert isinstance(result.op, packaging._parser.Op)
    assert isinstance(result.value, packaging._parser.Value)
    if result.variable.value == "extra":
        assert result.op.value == "=="
        return result, result.value.value
    else:
        return result, None

def marker_tree(marker: Marker | list[Any]) -> tuple[MarkerCombination | MarkerExpr, list[str]]:
    if isinstance(marker, Marker):
        markers = marker._markers
    else:
        markers = marker

    if isinstance(markers[0], list):
        previous, extras = marker_tree(markers[0])
    else:
        previous, e = marker_expr(markers[0])
        extras = []
        if e:
            extras.append(e)
    in_and = False
    result = []
    for i, expr in enumerate(markers[1:]):
        if i % 2 == 0:
            # 'and' or 'or'
            assert expr in ("and", "or")
            if expr == "or":
                result.append(previous)
                result.append(expr)
                in_and = False
            else:
                in_and = True
        else:
            n = expr
            if isinstance(n, list):
                n, es = marker_tree(n)
                extras.extend(es)
            if isinstance(n, tuple):
                n, e = marker_expr(n)
                if e:
                    extras.append(e)
            if in_and:
                previous = MarkerCombination(previous, "and", n)
            else:
                previous = n
    result.append(previous)

    previous = result[0]
    for i, expr in enumerate(result[1:]):
        if i % 2 == 1:
            previous = MarkerCombination(previous, "or", expr)

    return (previous, extras)

def invert_tree(tree: MarkerCombination | MarkerExpr):
    if isinstance(tree, MarkerExpr):
        assert tree.op.value not in ("not in", "in", "~=", "===")
        middle = {
            "<=": ">",
            "<": ">=",
            "!=": "==",
            ">=": "<",
            ">": "<=",
            "==": "!=",
        }[tree.op.value]
        return MarkerExpr(tree.variable, packaging._parser.Op(middle), tree.value)
    else:
        # combination
        reverse_combination = {"or": "and", "and": "or"}[tree.combination]
        return MarkerCombination(invert_tree(tree.left), reverse_combination, invert_tree(tree.right))

def tree_to_str(tree: MarkerCombination | MarkerExpr):
    if isinstance(tree, MarkerExpr):
        return f"{tree.variable.value} {tree.op.value} '{tree.value.value}'"
    else:
        return f"{tree_to_str(tree.left)} {tree.combination} {tree_to_str(tree.right)}"

variables = {
    "python_version": tuple(z3.Ints("python_version1 python_version2")),
    "os_name": z3.String("os_name"),
    "implementation_name": z3.String("implementation_name")
}

def tree_to_z3(tree: MarkerCombination | MarkerExpr):
    if isinstance(tree, MarkerCombination):
        if tree.combination == "and":
            return z3.And(tree_to_z3(tree.left), tree_to_z3(tree.right))
        else:
            assert tree.combination == "or"
            return z3.Or(tree_to_z3(tree.left), tree_to_z3(tree.right))
    else:
        var = variables[tree.variable.value]
        op = tree.op.value
        if tree.variable.value == "python_version":
            assert op in ("<=", "<", "==", "!=", ">", ">=")
            lop = {
                "<=": "<=",
                "<": "<=",
                "==": "==",
                ">": ">=",
                ">=": ">=",
                # ahahahaha
                "!=": ">= 0 *"
            }[op]
            r = False
            assert isinstance(tree.value.value, str)
            value = tuple(map(int, tree.value.value.split(".")))
            # I'm not really sure about this...
            if len(value) >= 2:
                r = z3.Or(eval(f"z3.And(z3_v[0] {lop} value[0], z3_v[1] {op} value[1])", {"z3": z3}, {"z3_v": var, "value": value}), r)
                op = {
                    ">=": ">",
                    "<=": "<"
                }.get(op, op)
            if len(value) >= 1:
                r = z3.Or(eval(f"z3.And(z3_v[0] {op} value[0])", {"z3": z3}, {"z3_v": var, "value": value}), r)
            assert len(value) in (1, 2)
            return r
        elif tree.variable.value == "os_name":
            assert op in ("==", "!=")
            return eval(f"z3_v {op} value", {"z3": z3}, {"z3_v": var, "value": tree.value.value})
        elif tree.variable.value == "implementation_name":
            assert op in ("==", "!=")
            return eval(f"z3_v {op} value", {"z3": z3}, {"z3_v": var, "value": tree.value.value})
        else:
            print(tree)
            assert False, tree.variable.value

def is_empty_marker(tree: MarkerCombination | MarkerExpr):
    s = z3.Solver()
    s.add(tree_to_z3(tree))
    result = s.check()
    assert result.r in (-1, 1)
    return result.r == -1

def parse_filename(filename: str, packagename: str) -> Version | None:
    if filename.endswith(".whl"):
        name, ver, _, _ = packaging.utils.parse_wheel_filename(filename)
    elif filename.endswith(".tar.gz"):
        name, ver = packaging.utils.parse_sdist_filename(filename)
    else:
        return None

    # TODO: bug report cffi?
    if name != packagename:
        return None

    assert name == packagename, f"{name} != {packagename} ; {filename}"
    return ver

package_cache = {}

async def package_info(package: str, c: httpx.AsyncClient):
    if package not in package_cache:
        package_cache[package] = (await c.get(f"https://pypi.org/simple/{package}/")).json()
    return package_cache[package]

version_cache = {}

async def release_info(package: str, version: Version, c: httpx.AsyncClient):
    if (package, version) not in version_cache:
        version_cache[(package, version)] = (await c.get(f"https://pypi.org/pypi/{package}/{version}/json", headers={"Accept": "application/json"})).json()
    return version_cache[(package, version)]

async def resolve(package: str, specifiers: SpecifierSet | None, markers: MarkerCombination | MarkerExpr | None, c: httpx.AsyncClient):
    # TODO: hashes
    name = packaging.utils.canonicalize_name(package, validate=True)
    info = await package_info(name, c)

    files = []
    for file in info["files"]:
        version = parse_filename(file["filename"], name)
        if version is not None:
            files.append((version, file))

    if specifiers:
        files = list(filter(lambda f: f[0] in specifiers, files))
    files.sort(key=lambda f: (-f[0].is_prerelease, f[0]), reverse=True)

    versions = collections.defaultdict(lambda: [])
    for file in files:
        versions[file[0]].append(file[1])

    # TODO: how to handle markers? im not even sure the behavior
    results = []
    if not markers:
        results.append((None, next(iter(versions))))

    for version, files in versions.items():
        if not markers or is_empty_marker(markers):
            break

        # I think this only needs to handle python_version?
        # assumption: there's only one `requires-python` key value
        file = files[0]
        if "requires-python" in file and file["requires-python"]:
            previous = None
            var = packaging._parser.Variable("python_version")
            for spec in SpecifierSet(file["requires-python"]):
                op = packaging._parser.Op(spec.operator)
                value = packaging._parser.Value(spec.version)
                n = MarkerExpr(var, op, value)
                if previous is None:
                    previous = n
                else:
                    previous = MarkerCombination(previous, "and", n)

            assert previous
            combined = MarkerCombination(
                previous, "and", markers
            )
            if not is_empty_marker(combined):
                results.append((combined, version))
                markers = MarkerCombination(invert_tree(previous), "and", markers)
        else:
            results.append((markers, version))
            markers = None
            break

    if markers and not is_empty_marker(markers) or not results:
        assert False, "no candidates found"

    return results

async def lock_deps(package: str, version: Version, markers: MarkerCombination | MarkerExpr | None, c: httpx.AsyncClient, extras: list[str]):
    results = []
    r = await release_info(package, version, c)
    async def temp(pkg: str, specifiers: SpecifierSet | None, markers: MarkerCombination | MarkerExpr | None):
        results.extend(await lock(pkg, specifiers, markers, c, []))

    async with trio.open_nursery() as nursery:
        if not r["info"]["requires_dist"]:
            return results

        for dependency in r["info"]["requires_dist"]:
            dep = Requirement(dependency)

            dep_markers = markers
            if dep.marker:
                ms, es = marker_tree(dep.marker)
                if es:
                    # TODO: this is technically incorrect...
                    # (think "extra == 'a' and extra == 'b'")
                    for extra in extras:
                        if extra in es:
                            break
                    else:
                        continue
                dep_markers = MarkerCombination(ms, "and", dep_markers)

            s = z3.Solver()
            s.add(tree_to_z3(dep_markers))
            if s.check() == -1:
                continue

            assert dep.url is None
            assert not dep.extras, dep
            nursery.start_soon(temp, dep.name, dep.specifier, dep_markers)

    return results

async def lock(package: str, specifiers: SpecifierSet | None, markers: MarkerCombination | MarkerExpr | None, c: httpx.AsyncClient, extras: list[str]):
    results = []
    unprocessed_results = [(package, *a) for a in await resolve(
        package,
        specifiers,
        markers,
        c
    )]
    t = []
    async def temp(pkg: str, version: Version, markers: MarkerCombination | MarkerExpr | None, c: httpx.AsyncClient):
        t.extend(await lock_deps(package, version, markers, c, extras))

    async with trio.open_nursery() as nursery:
        for pkg, markers, version in unprocessed_results:
            nursery.start_soon(temp, pkg, version, markers, c)
            results.append((pkg, markers, version))

    return results + t

async def main() -> None:
    async with httpx.AsyncClient(headers={"Accept": "application/vnd.pypi.simple.v1+json"}) as c:
        r = await lock(
            "trio",
            None,
            marker_tree(Marker("python_version >= '3.6'"))[0],
            c,
            []
        )
        for pkg, markers, version in r:
            # TODO: simplify?
            print(f"{pkg}=={version} ; {tree_to_str(markers)}")

trio.run(main)
