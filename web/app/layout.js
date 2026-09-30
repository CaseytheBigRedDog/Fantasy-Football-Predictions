import "./globals.css";

export const metadata = {
  title: "Fantasy Football Predictions",
  description: "Weekly full-PPR projections with a start/sit assistant grounded in my own model.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
