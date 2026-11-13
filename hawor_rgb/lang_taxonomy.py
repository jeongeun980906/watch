"""Canonical activity taxonomy for the egoverse demo<->query pairing GATE.
Replaces the fragile keyword coarse()/fine() guessing with the dataset's own language tags.

KEY DATA FACTS (scanned 13457 eps, see docs/egoverse_pairing_v9_log.md):
 - aria_bimanual (657) and scale (6086) carry a CLEAN `task_name` attr -> strip
   freeform_/flagship_ prefix and _success/_in_domain/_static_cam suffix -> canonical.
 - MECKA_BIMANUAL (6714) has task_name == "debug" for ALL episodes (placeholder, NOT junk!).
   The real activity is ONLY in `task_description` -> classify by keywords (98% coverage).
 - `annotations` array is empty -> no per-frame subgoal labels exist.

Use: activity(emb, task_name, task_desc) -> canonical str. SUPER[canonical] -> cross-emb
fallback super-group (used only when an exact canonical has no cross-emb member).
"""
CONT=['container','box','pouch','plastic','tray','case','bag','carton','packet','jar','bottle',
      'compartment','dish','plate','crate','bowl','cup','glass']

_TN={
 "cup_on_saucer":"cup_on_saucer","put_cup_on_saucer":"cup_on_saucer",
 "bag_groceries":"bagging_groceries","bagging_groceries":"bagging_groceries",
 "unpack_grocery_bag":"unpack_grocery","unpack_groceries":"unpack_grocery",
 "unpack_grocery_bag_and_put_items_into_fridge":"unpack_grocery",
 "put_object_in_container":"put_object_in_container","object_in_container":"put_object_in_container",
 "put_scissors_and_color_box_into_the_plastic_containers":"put_object_in_container",
 "do_dishes":"do_dishes","clean_up_the_dishes":"do_dishes",
 "load_dishwasher":"load_dishwasher","unload_dishwasher":"unload_dishwasher","load_unload_dishwasher":"load_dishwasher",
 "return_small_objects_to_bins":"sort_bins",
 "tidy_up_kitchen":"tidy_kitchen","placing_utensils_into_a_drawer":"utensils_in_drawer",
 "pick_place":"pick_place","stack_lego_bricks":"stack","stack_coins":"stack",
}
def _canon_taskname(tn):
    s=tn.lower()
    for p in ("freeform_","flagship_"):
        if s.startswith(p): s=s[len(p):]
    for suf in ("_success_3","_success_2","_success","_in_domain","_indomain","_static_cam"):
        if s.endswith(suf): s=s[:-len(suf)]
    return _TN.get(s,s)

def _canon_desc(t):  # MECKA: classify task_description -> SAME canonical names as task_name path
    t=t.lower()
    if 'dishwasher' in t: return 'unload_dishwasher' if 'unload' in t else ('load_dishwasher' if 'load' in t else 'dishwasher_other')
    if any(k in t for k in ['wash','scrub','rinse','soapy','do the dishes']) and any(k in t for k in ['dish','plate','bowl','pan','pot','cookware','utensil','sink','sponge','cup','glass']): return 'do_dishes'
    if 'fold' in t: return 'fold'
    if 'tie' in t and ('lace' in t or 'shoe' in t): return 'tie'
    if 'stack' in t: return 'stack'
    if any(k in t for k in ['thread','bead','bracelet','assembl','attach','connect','build']): return 'assemble'
    if any(k in t for k in ['sort','separat','categoriz']) or (('organiz' in t or 'arrang' in t) and any(k in t for k in CONT+['bin','group','type','color'])): return 'sort_bins'
    if any(k in t for k in ['scoop','pour','fill','spoon','ladle','funnel','portion','weigh','syringe','dispens']): return 'fill_scoop'
    if any(k in t for k in ['seal','tape','wrap','zip','adhesive','heat-seal']): return 'seal_wrap'
    if 'unpack' in t or ('remov' in t and 'bag' in t) or ('taking' in t and 'bag' in t): return 'unpack_grocery'
    if 'grocer' in t and 'bag' in t: return 'bagging_groceries'
    if any(k in t for k in ['pack','packag','plac','put','insert','load','bagging','bag ','gather','deposit','place']) and any(k in t for k in CONT): return 'put_object_in_container'
    if any(k in t for k in ['wip','clean','dust','polish','smooth','inspect']): return 'clean_wipe'
    if any(k in t for k in ['pick','plac']): return 'pick_place'
    return 'other'

def activity(emb, task_name, task_desc):
    return _canon_desc(task_desc) if emb.split("_")[0]=="MECKA" else _canon_taskname(task_name)

# cross-emb fallback super-groups (used only when the exact canonical has no cross-emb member).
# opposites kept apart on purpose: bagging != unpack ; load != unload.
SUPER={
 'cup_on_saucer':'cup',
 'bagging_groceries':'packaging','put_object_in_container':'packaging','fill_scoop':'packaging',
 'seal_wrap':'packaging','utensils_in_drawer':'packaging',
 'unpack_grocery':'unpack',
 'do_dishes':'dishes',                       # dishwasher load/unload share the 'dishes' super so a
 'load_dishwasher':'dishes','unload_dishwasher':'dishes','dishwasher_other':'dishes',  # dishwasher query's cross-fallback finds a dish-washing scene
 'sort_bins':'sorting',
 'stack':'stack','fold':'fold','assemble':'assemble','tie':'assemble',
 'clean_wipe':'cleaning','tidy_kitchen':'cleaning',
 'pick_place':'pickplace','other':'other',
}
def super_of(canon): return SUPER.get(canon, canon)
