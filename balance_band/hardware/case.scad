// Balance Band — waist case, parametric FIRST DRAFT (2026-09-23)
// NOT rendered or test-printed. A starting point for the final design (Claude Design).
//
// Worn on a belt over the lower back (L3–L5). Frame used here, seen from BEHIND the person:
//   +X = along the belt, toward the person's RIGHT      (so the person's LEFT is -X)
//   +Y = UP
//   +Z = away from the body (z = 0 is the back plate that touches the person)
//
// Two parts:
//   base : back plate + belt tunnel + body with battery bay, board bay, switch slot, USB-C opening
//   lid  : closes the body; 2 x M2 screws; LED window; engraved "UP" and "L" arrows
//
// Inside: the LiPo lies in the lower bay, the XIAO board lies FLAT on the floor of the upper bay
// (so its IMU is flat and rigid), USB-C pointing UP through the top wall.

part = "all";          // "base", "lid" or "all" (assembled view)
$fn = 40;

// ---------------- parts that go inside (measure yours and adjust) ----------------
bat = [35, 25, 5.5];   // LiPo "502535": X (along belt), Y (up), Z (thickness) incl. protection board
brd = [17.8, 21, 1.2]; // Seeed XIAO nRF52840 Sense PCB: X, Y, thickness; USB-C on its +Y (top) edge
brd_parts_h = 3.3;     // tallest part on top of the board (the USB-C socket)
usb = [10, 4.5];       // USB-C opening in the top wall: width (X), height (Z) above the floor
sw  = [9, 4];          // slide-switch slot in the left (-X) wall: length (Y), height (Z)
led_d = 2.5;           // LED window diameter in the lid
led_off = [0, -4];     // LED position relative to the board's top-centre edge (X, Y)

// ---------------- case ----------------
wall = 2;              // side walls
floor_t = 1.6;         // floor of the body
lid_t = 1.6;           // lid plate
lip_h = 2;             // lid lip that drops into the body
lip_t = 1.2;           // lid lip wall thickness
clr = 0.4;             // fit clearance
gap_y = 5;             // band between battery bay and board bay (holds the two screw bosses)
boss_d = 4.5;          // screw boss diameter
screw_pilot = 1.8;     // pilot hole for M2 self-tapping screws
screw_clear = 2.4;     // clearance hole in the lid
engrave = 0.6;         // depth of the engraved arrows / letters

// ---------------- belt ----------------
belt_w = 40;           // belt (webbing) width. 40 mm non-stretch webbing
belt_gap = 4;          // belt thickness + slack; the tunnel the belt passes through
back_t = 2.4;          // back plate (touches the person)

// ---------------- derived ----------------
inner = [bat[0] + 2*clr,
         bat[1] + gap_y + brd[1] + 3*clr,
         max(bat[2], brd[2] + brd_parts_h) + 1];
outer = [inner[0] + 2*wall, inner[1] + 2*wall, floor_t + inner[2]];
z0 = back_t + belt_gap;                         // bottom of the body
bridge = (outer[1] - belt_w - 2*clr) / 2;       // top/bottom bridges that close the belt tunnel
assert(bridge >= 3, "belt too wide for this case height: raise inner Y or use narrower webbing");

bat_y0 = wall + clr;                            // battery bay starts
boss_y = wall + bat[1] + 2*clr + gap_y/2;       // screw bosses sit in the band between bays
boss_xy = [[wall + boss_d/2, boss_y], [outer[0] - wall - boss_d/2, boss_y]];
brd_x0 = outer[0]/2 - brd[0]/2;                 // board centred along X
brd_y0 = outer[1] - wall - clr - brd[1];        // board pushed to the top wall (USB-C at the opening)

module base() {
    difference() {
        union() {
            translate([0, 0, z0]) cube(outer);                      // body
            cube([outer[0], outer[1], back_t]);                     // back plate
            cube([outer[0], bridge, z0]);                           // bottom bridge of belt tunnel
            translate([0, outer[1] - bridge, 0]) cube([outer[0], bridge, z0]);   // top bridge
        }
        translate([wall, wall, z0 + floor_t]) cube([inner[0], inner[1], inner[2] + 1]);   // cavity
        translate([outer[0]/2 - usb[0]/2, outer[1] - wall - 1, z0 + floor_t])             // USB-C
            cube([usb[0], wall + 2, usb[1]]);
        translate([-1, bat_y0 + bat[1]/2 - sw[0]/2, z0 + floor_t + 1])                     // switch
            cube([wall + 2, sw[0], sw[1]]);
    }
    // screw bosses
    for (p = boss_xy) translate([p[0], p[1], z0 + floor_t])
        difference() {
            cylinder(d = boss_d, h = inner[2]);
            translate([0, 0, 1]) cylinder(d = screw_pilot, h = inner[2]);
        }
    // low frame that locates the board (open at the top for the USB-C); fix board with foam tape
    translate([0, 0, z0 + floor_t])
        difference() {
            translate([brd_x0 - clr - 1, brd_y0 - clr - 1, 0]) cube([brd[0] + 2*clr + 2, brd[1] + clr + 1, 2]);
            translate([brd_x0 - clr, brd_y0 - clr, -1]) cube([brd[0] + 2*clr, brd[1] + 2*clr + 2, 4]);
        }
}

module arrow(len = 12, shaft = 2.4, head = 6) {       // points +Y, starts at origin
    polygon([[-shaft/2, 0], [shaft/2, 0], [shaft/2, len - head], [head/2, len - head],
             [0, len], [-head/2, len - head], [-shaft/2, len - head]]);
}

module lid() {
    difference() {
        union() {
            cube([outer[0], outer[1], lid_t]);
            translate([wall + clr, wall + clr, -lip_h])
                difference() {
                    cube([inner[0] - 2*clr, inner[1] - 2*clr, lip_h]);
                    translate([lip_t, lip_t, -1]) cube([inner[0] - 2*clr - 2*lip_t, inner[1] - 2*clr - 2*lip_t, lip_h + 2]);
                }
        }
        for (p = boss_xy) {
            translate([p[0], p[1], -lip_h - 1]) cylinder(d = screw_clear, h = lid_t + lip_h + 2);
            translate([p[0], p[1], -lip_h - 1]) cylinder(d = boss_d + 2*clr, h = lip_h + 1);   // lip clears bosses
        }
        translate([outer[0]/2 + led_off[0], outer[1] - wall - clr + led_off[1], -lip_h - 1])       // LED window
            cylinder(d = led_d, h = lid_t + lip_h + 2);
        // engraved marks on the outer face: UP arrow (+Y) and L arrow (person's left = -X)
        translate([0, 0, lid_t - engrave]) linear_extrude(engrave + 1) {
            translate([outer[0]/2, outer[1]*0.30]) arrow();
            translate([outer[0]/2, outer[1]*0.30 - 7]) text("UP", size = 5, halign = "center",
                                                               font = "Liberation Sans:style=Bold");
            translate([outer[0]/2 + 8, outer[1]*0.12]) rotate(90) arrow(len = 10);
            translate([outer[0]/2 - 11, outer[1]*0.12 - 2.5]) text("L", size = 5, halign = "center",
                                                                   font = "Liberation Sans:style=Bold");
        }
    }
}

if (part == "base") base();
if (part == "lid") translate([0, 0, lid_t]) mirror([0, 0, 1]) lid();     // outer face down on the bed
if (part == "all") { base(); translate([0, 0, z0 + outer[2]]) lid(); }

echo(str("Outer size (X along belt, Y up, Z depth incl. belt tunnel): ",
         outer[0], " x ", outer[1], " x ", z0 + outer[2] + lid_t, " mm"));
